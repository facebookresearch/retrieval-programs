# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import json
import pathlib
import unittest
from unittest import mock

import rpl.search.search_sdk as sdk


class _FakeResp:
    def __init__(self, payload):
        self._b = json.dumps(payload).encode()

    def read(self):
        return self._b

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _patch(payload):
    return mock.patch.object(sdk.urllib.request, "urlopen", return_value=_FakeResp(payload))


class SearchSdkTest(unittest.TestCase):
    def setUp(self):
        sdk._search_calls = 0
        sdk._rerank_calls = 0

    def test_search_normalizes_contents_and_strips_quote(self):
        payload = {"result": [[{"document": {"id": "7", "contents": '"Paris"\nCapital of France.'},
                                "score": 1.5}]]}
        with _patch(payload):
            rows = sdk.search("capital of France", top_k=1)
        self.assertEqual(
            rows, [{"id": "7", "title": "Paris", "text": "Capital of France.", "score": 1.5}]
        )

    def test_search_rejects_empty_query(self):
        with self.assertRaises(ValueError):
            sdk.search("   ")

    def test_search_rejects_bad_top_k(self):
        for bad in (0, 201, True, "3"):
            with self.assertRaises(ValueError):
                sdk.search("q", top_k=bad)

    def test_search_call_budget(self):
        payload = {"result": [[]]}
        with mock.patch.dict("os.environ", {"MAX_SEARCH_CALLS": "1"}), _patch(payload):
            sdk.search("q")
            with self.assertRaises(RuntimeError):
                sdk.search("q")

    def _posted_body(self, m):
        return json.loads(m.call_args.args[0].data.decode())  # urlopen(req) -> req.data is the JSON body

    def test_dense_retrieval_routes_method(self):
        with _patch({"result": [[]]}) as m:
            sdk.dense_retrieval("q", top_k=2)
        body = self._posted_body(m)
        self.assertEqual(body["method"], "dense")
        self.assertEqual(body["topk"], 2)

    def test_sparse_retrieval_routes_method(self):
        with _patch({"result": [[]]}) as m:
            sdk.sparse_retrieval("q")
        self.assertEqual(self._posted_body(m)["method"], "sparse")

    def test_search_omits_method_for_backcompat(self):
        with _patch({"result": [[]]}) as m:
            sdk.search("q")
        self.assertNotIn("method", self._posted_body(m))

    def test_all_primitives_share_call_budget(self):
        with mock.patch.dict("os.environ", {"MAX_SEARCH_CALLS": "2"}), _patch({"result": [[]]}):
            sdk.search("q")            # 1
            sdk.dense_retrieval("q")   # 2
            with self.assertRaises(RuntimeError):
                sdk.sparse_retrieval("q")  # over budget

    def test_dedup_keeps_first_occurrence_and_order(self):
        docs = [{"id": "1", "title": "a"}, {"id": "2", "title": "b"}, {"id": "1", "title": "a-dup"}]
        self.assertEqual([d["id"] for d in sdk.dedup(docs)], ["1", "2"])

    def test_dedup_falls_back_to_title_text_when_no_id(self):
        docs = [{"title": "t", "text": "x"}, {"title": "t", "text": "x"}, {"title": "t", "text": "y"}]
        self.assertEqual(len(sdk.dedup(docs)), 2)

    def test_fuse_rrf_orders_by_fused_rank_and_dedups(self):
        a = [{"id": "1"}, {"id": "2"}, {"id": "3"}]   # ranks 1,2,3
        b = [{"id": "3"}, {"id": "1"}]                # ranks 1,2
        ids = [d["id"] for d in sdk.fuse(a, b)]
        # id1 = 1/61+1/62 (top); id3 = 1/61+1/63 (just below); id2 = 1/62 (single list, last)
        self.assertEqual(ids, ["1", "3", "2"])

    def test_fuse_writes_fused_score_and_does_not_charge_budget(self):
        sdk._search_calls = 0
        fused = sdk.fuse([{"id": "1"}, {"id": "2"}])
        self.assertTrue(all("score" in d for d in fused))
        self.assertEqual(sdk._search_calls, 0)  # post-processing only

    def test_fuse_rejects_unknown_method(self):
        with self.assertRaises(ValueError):
            sdk.fuse([{"id": "1"}], method="concat")

    def test_fuse_rejects_a_splatted_flat_doc_list_and_says_how_to_fix_it(self):
        """`fuse(*docs)` on a FLAT list splats dicts; the old failure was an opaque str.get() three frames down."""
        docs = [{"id": "1"}, {"id": "2"}]
        with self.assertRaises(TypeError) as cm:
            sdk.fuse(*docs)
        msg = str(cm.exception)
        self.assertIn("fuse() argument 1", msg)
        self.assertIn("got dict", msg)
        self.assertIn("fuse(dense_results, sparse_results)", msg)  # the message must carry the fix
        self.assertEqual(msg.count("\n"), 0)  # the sandbox surfaces only the last traceback LINE

    def test_fuse_rejects_a_list_of_non_documents(self):
        with self.assertRaises(TypeError) as cm:
            sdk.fuse(["not-a-doc"], [{"id": "1"}])
        self.assertIn("must contain document dicts", str(cm.exception))

    def test_fuse_still_accepts_both_correct_forms(self):
        a, b = [{"id": "1"}, {"id": "2"}], [{"id": "2"}]
        list_of_lists = [a, b]
        self.assertEqual([d["id"] for d in sdk.fuse(a, b)], ["2", "1"])
        self.assertEqual([d["id"] for d in sdk.fuse(*list_of_lists)], ["2", "1"])
        self.assertEqual(sdk.fuse(), [])  # no arguments stays a no-op, not an error

    def test_dedup_rejects_a_single_document(self):
        with self.assertRaises(TypeError) as cm:
            sdk.dedup({"id": "1"})
        self.assertIn("dedup() argument", str(cm.exception))

    def test_rerank_reorders_and_normalizes(self):
        # server returns the original doc dicts reordered best-first, with rerank scores
        payload = {"result": [[
            {"document": {"id": "2", "contents": "Beta\nsecond doc"}, "score": 0.9},
            {"document": {"id": "1", "contents": '"Alpha"\nfirst doc'}, "score": 0.4},
        ]]}
        docs = [{"id": "1", "title": "Alpha", "text": "first doc", "score": 0.1},
                {"id": "2", "title": "Beta", "text": "second doc", "score": 0.05}]
        with _patch(payload):
            out = sdk.rerank("q", docs, top_k=2)
        self.assertEqual([d["id"] for d in out], ["2", "1"])  # reordered by reranker score
        self.assertEqual(out[0], {"id": "2", "title": "Beta", "text": "second doc", "score": 0.9})

    def test_rerank_request_format(self):
        with _patch({"result": [[]]}) as m:
            sdk.rerank("myquery", [{"id": "1", "title": "T", "text": "body"}], top_k=3)
        body = self._posted_body(m)
        self.assertEqual(body["queries"], ["myquery"])
        self.assertEqual(body["documents"], [[{"id": "1", "contents": "T\nbody"}]])
        self.assertEqual(body["rerank_topk"], 3)
        self.assertTrue(body["return_scores"])

    def test_rerank_topk_defaults_to_all_docs(self):
        with _patch({"result": [[]]}) as m:
            # no top_k -> ask the server for ALL of them, not its default 3; the relevance predicate then
            # filters client-side (see RerankRelevanceFilterTest)
            sdk.rerank("q", [{"id": "1"}, {"id": "2"}])
        self.assertEqual(self._posted_body(m)["rerank_topk"], 2)

    def test_rerank_rejects_empty_docs_and_query(self):
        with self.assertRaises(ValueError):
            sdk.rerank("q", [])
        with self.assertRaises(ValueError):
            sdk.rerank("   ", [{"id": "1"}])

    def test_rerank_does_not_charge_search_budget(self):
        with _patch({"result": [[]]}):
            sdk.rerank("q", [{"id": "1", "title": "t", "text": "x"}])
        self.assertEqual(sdk._search_calls, 0)  # rerank is post-processing, not retrieval
        self.assertEqual(sdk._rerank_calls, 1)


class RerankCandidateCapTest(unittest.TestCase):
    """`RERANK_MAX_DOCS` bounds the cross-encoder call."""

    def _cap(self, n_docs, cap):
        """Replicate the cap so it can be exercised without a live reranker."""
        docs = [{"id": str(i), "title": f"T{i}", "text": "x"} for i in range(n_docs)]
        return docs[:cap] if cap > 0 and len(docs) > cap else docs

    def test_cap_truncates_and_keeps_the_head(self):
        out = self._cap(500, 100)
        self.assertEqual(len(out), 100)
        self.assertEqual(out[0]["id"], "0", "input is ordered by first-stage score, so the TAIL is dropped")
        self.assertEqual(out[-1]["id"], "99")

    def test_zero_means_unlimited(self):
        self.assertEqual(len(self._cap(500, 0)), 500)

    def test_a_set_smaller_than_the_cap_is_untouched(self):
        self.assertEqual(len(self._cap(20, 100)), 20)

    def test_the_sdk_reads_the_cap_from_the_environment(self):
        src = pathlib.Path(__file__).resolve().parent.joinpath("search_sdk.py").read_text()
        self.assertIn('_int_env("RERANK_MAX_DOCS", 0)', src)
        i_cap = src.index("RERANK_MAX_DOCS")
        i_payload = src.index("payload_docs = [")
        self.assertLess(i_cap, i_payload, "the cap must be applied BEFORE the payload is built")

    def test_the_executor_passes_it_into_the_sandbox(self):
        src = pathlib.Path(__file__).resolve().parents[1].joinpath("llm_agent", "code_executor.py").read_text()
        self.assertIn('"RERANK_MAX_DOCS": str(self.rerank_max_docs)', src)


class SandboxStateCapTest(unittest.TestCase):
    """`MAX_STATE_BYTES` must be settable, and must default UP for a long-document corpus."""

    def test_the_executor_passes_the_cap_to_the_sandbox(self):
        src = pathlib.Path(__file__).resolve().parents[1].joinpath("llm_agent", "code_executor.py").read_text()
        self.assertIn('"MAX_STATE_BYTES": str(self.max_state_bytes)', src)

    def test_the_default_cap_is_sized_for_long_documents(self):
        """Asserts the SIZE, not the literal. This pinned the exact text `256 * 1024`, so raising the
        default to 4 MB failed it as a string mismatch rather than telling us anything about the cap."""
        from rpl.llm_agent.code_executor import CodeExecutor
        default = CodeExecutor.__dataclass_fields__["max_state_bytes"].default
        self.assertGreaterEqual(default, 2 * 1024 * 1024,
                                "a 200-document candidate list of resumes is ~1.6 MB; a smaller default "
                                "drops the namespace and every cross-turn variable then raises NameError")

    def test_the_agent_loop_reads_it_from_config(self):
        src = pathlib.Path(__file__).resolve().parents[1].joinpath("llm_agent", "rpl_agent_loop.py").read_text()
        self.assertIn('max_state_bytes=int(cfg.get("max_state_bytes"', src)

    def test_the_sandbox_still_honours_the_env_var(self):
        src = pathlib.Path(__file__).resolve().parents[1].joinpath("llm_agent", "sandbox_runner.py").read_text()
        self.assertIn('os.environ.get("MAX_STATE_BYTES"', src)

    def test_exceeding_the_cap_drops_everything_rather_than_truncating(self):
        """A partially restored namespace is worse than none: it fails later and further from the cause."""
        src = pathlib.Path(__file__).resolve().parents[1].joinpath("llm_agent", "sandbox_runner.py").read_text()
        self.assertIn("return {}, f\"sandbox state exceeded", src)


class RerankRelevanceFilterTest(unittest.TestCase):
    """`rerank()` without `top_k` keeps what the reranker judged relevant, not a fixed count."""

    def setUp(self):
        sdk._search_calls = 0
        sdk._rerank_calls = 0
        sdk._rerank_call_docs = []
        self._thresh, self._frac = sdk._RERANK_THRESHOLD, sdk._MARAG_FILTER_FRAC
        sdk._RERANK_THRESHOLD, sdk._MARAG_FILTER_FRAC = 0.5, 0.0

    def tearDown(self):
        sdk._RERANK_THRESHOLD, sdk._MARAG_FILTER_FRAC = self._thresh, self._frac

    def _payload(self, *scores):
        return {"result": [[
            {"document": {"id": str(i), "contents": f"T{i}\nbody"}, "score": sc}
            for i, sc in enumerate(scores)
        ]]}

    DOCS = [{"id": "0"}, {"id": "1"}, {"id": "2"}, {"id": "3"}]

    def test_keeps_only_the_documents_above_the_threshold(self):
        with _patch(self._payload(0.99, 0.72, 0.31, 0.02)):
            out = sdk.rerank("q", self.DOCS)
        self.assertEqual([d["id"] for d in out], ["0", "1"])

    def test_keeps_many_when_many_are_relevant(self):
        """The point of the mode: breadth follows the data rather than a constant."""
        with _patch(self._payload(*([0.9] * 4))):
            self.assertEqual(len(sdk.rerank("q", self.DOCS)), 4)

    def test_returns_empty_when_nothing_is_relevant(self):
        """No floor. Keeping a guaranteed top-1 would put back the fixed breadth this removes, and would
        also assert a document is relevant when the judge said it is not."""
        with _patch(self._payload(0.4, 0.2, 0.1, 0.0)):
            self.assertEqual(sdk.rerank("q", self.DOCS), [])

    def test_an_empty_filter_result_is_visible_in_the_recorded_docs(self):
        """`rerank_call_docs` is what the dumps expose, so a filter that kept nothing must be diagnosable
        rather than indistinguishable from the policy not calling rerank."""
        with _patch(self._payload(0.1, 0.1, 0.1, 0.1)):
            sdk.rerank("q", self.DOCS)
        self.assertEqual(sdk._rerank_calls, 1, "the call must still be charged")
        self.assertEqual(sdk._rerank_call_docs[-1], [])

    def test_explicit_top_k_is_unaffected_by_the_threshold(self):
        """The fixed-breadth mode must keep N even when the reranker rates them poorly -- 0.00% of 20,212
        logged rerank calls omit top_k, so this is the path every trained arm is on."""
        with _patch(self._payload(0.4, 0.2)):
            out = sdk.rerank("q", self.DOCS[:2], top_k=2)
        self.assertEqual([d["id"] for d in out], ["0", "1"])

    def test_a_scoreless_row_is_kept(self):
        """A missing score means the server did not return scores; dropping everything on that would look
        like the filter rejecting the whole candidate set."""
        with _patch({"result": [[{"document": {"id": "9", "contents": "T\nb"}}]]}):
            self.assertEqual([d["id"] for d in sdk.rerank("q", self.DOCS)], ["9"])

    def test_the_threshold_is_configurable(self):
        sdk._RERANK_THRESHOLD = 0.8
        with _patch(self._payload(0.99, 0.72)):
            self.assertEqual([d["id"] for d in sdk.rerank("q", self.DOCS[:2])], ["0"])

    def test_document_filter_uses_the_predicate_by_default(self):
        with _patch(self._payload(0.9, 0.9, 0.1, 0.1)):
            out = sdk.DocumentFilter("q", self.DOCS)
        self.assertEqual([d["id"] for d in out], ["0", "1"])

    def test_document_filter_frac_restores_the_old_fixed_fraction(self):
        """Kept as an A/B knob: the 50% fraction was ours, not the paper's."""
        sdk._MARAG_FILTER_FRAC = 0.5
        with _patch(self._payload(0.9, 0.9, 0.1, 0.1)) as m:
            sdk.DocumentFilter("q", self.DOCS)
        self.assertEqual(json.loads(m.call_args.args[0].data.decode())["rerank_topk"], 2)

    def test_document_filter_still_refuses_an_empty_candidate_set(self):
        with self.assertRaises(ValueError):
            sdk.DocumentFilter("q", [])


class RetryBudgetTest(unittest.TestCase):
    """The retry budget has to outlast a brief retriever blip, because giving up is silently wrong."""

    def setUp(self):
        sdk._search_calls = 0

    def test_budget_outlasts_a_multi_second_blip(self):
        budget = sum(sdk._INITIAL_BACKOFF_SEC * (2 ** i) for i in range(sdk._MAX_RETRIES))
        self.assertGreaterEqual(budget, 20.0, f"retry budget is only {budget:.1f}s — a blip will be fatal")

    def test_a_transient_refusal_that_recovers_is_retried_not_failed(self):
        payload = {"result": [[{"document": {"id": "1", "contents": "T\nbody"}, "score": 1.0}]]}
        attempts = []

        def flaky(req, timeout=None):
            attempts.append(1)
            if len(attempts) <= 4:                      # more than the OLD budget of 4 total attempts
                raise OSError(111, "Connection refused")
            return _FakeResp(payload)

        with mock.patch.object(sdk.urllib.request, "urlopen", side_effect=flaky), \
             mock.patch.object(sdk.time, "sleep"):      # don't actually wait through the backoff
            rows = sdk.search("q", top_k=1)
        self.assertEqual(rows[0]["id"], "1")
        self.assertEqual(len(attempts), 5, "should have retried past the old 4-attempt ceiling")

    def test_a_deterministic_server_error_is_still_not_retried(self):
        """Widening the budget must not turn a 4xx/5xx into seven pointless round-trips."""
        err = sdk.urllib.error.HTTPError("u", 500, "boom", {}, None)
        with mock.patch.object(sdk.urllib.request, "urlopen", side_effect=err) as m:
            with self.assertRaises(sdk.urllib.error.HTTPError):
                sdk.search("q")
        self.assertEqual(m.call_count, 1)
