# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for the rerank server's (query, doc) -> score LRU."""

import threading
import unittest

from rpl.search.rerank_server import BaseCrossEncoder


class _CountingEncoder(BaseCrossEncoder):
    """Scores a pair as len(query) + len(doc) and counts how many pairs actually reached the model."""

    def __init__(self):
        self.forwarded = []

    def _predict(self, pairs):
        self.forwarded.extend(pairs)
        return [float(len(q) + len(d)) for q, d in pairs]

    def _passage_to_string(self, doc_item):
        return doc_item["contents"]


def _docs(*names):
    return [{"id": n, "contents": n} for n in names]


class RerankCacheTest(unittest.TestCase):
    def test_disabled_cache_is_passthrough(self):
        enc = _CountingEncoder()
        out = enc.rerank(["ab"], [_docs("xxx", "y")])
        self.assertEqual([s for _d, s in out[0]], [5.0, 3.0])  # sorted best-first
        self.assertEqual(len(enc.forwarded), 2)
        self.assertEqual(enc.cache_stats()["entries"], 0)

    def test_scores_identical_with_and_without_cache(self):
        plain, cached = _CountingEncoder(), _CountingEncoder()
        cached.enable_cache(1000)
        docs = _docs("alpha", "bb", "cccc")
        want = {d["id"]: s for d, s in plain.rerank(["query"], [docs])[0]}
        got = {d["id"]: s for d, s in cached.rerank(["query"], [docs])[0]}
        self.assertEqual(want, got)

    def test_repeated_call_hits_cache(self):
        enc = _CountingEncoder()
        enc.enable_cache(1000)
        docs = _docs("aa", "bb")
        first = enc.rerank(["q"], [docs])
        self.assertEqual(len(enc.forwarded), 2)
        second = enc.rerank(["q"], [docs])  # identical GRPO sibling rollout
        self.assertEqual(len(enc.forwarded), 2, "second identical call must not reach the model")
        self.assertEqual([s for _d, s in first[0]], [s for _d, s in second[0]])
        st = enc.cache_stats()
        self.assertEqual((st["pairs"], st["hits"], st["forwards"]), (4, 2, 2))
        self.assertEqual(st["dedup_x"], 2.0)

    def test_duplicate_pairs_within_one_request_collapse(self):
        enc = _CountingEncoder()
        enc.enable_cache(1000)
        # Same doc reached via two queries' candidate lists, plus a repeat inside one list (fused pools).
        enc.rerank(["q", "q"], [_docs("dup", "dup", "other"), _docs("dup")])
        self.assertEqual(sorted(set(enc.forwarded)), sorted({("q", "dup"), ("q", "other")}))
        self.assertEqual(len(enc.forwarded), 2, "duplicate pairs must be forwarded once")

    def test_distinct_query_or_doc_is_a_miss(self):
        enc = _CountingEncoder()
        enc.enable_cache(1000)
        enc.rerank(["q1"], [_docs("d")])
        enc.rerank(["q2"], [_docs("d")])
        enc.rerank(["q1"], [_docs("d2")])
        self.assertEqual(len(enc.forwarded), 3)

    def test_lru_evicts_oldest(self):
        enc = _CountingEncoder()
        enc.enable_cache(2)
        for name in ("a", "b", "c"):  # 3 distinct pairs into a size-2 cache
            enc.rerank(["q"], [_docs(name)])
        self.assertEqual(len(enc.forwarded), 3)
        self.assertEqual(enc.cache_stats()["entries"], 2)
        enc.rerank(["q"], [_docs("a")])  # "a" was evicted -> re-forwarded
        self.assertEqual(len(enc.forwarded), 4)
        enc.rerank(["q"], [_docs("c")])  # "c" still resident
        self.assertEqual(len(enc.forwarded), 4)

    def test_zero_size_leaves_cache_disabled(self):
        enc = _CountingEncoder()
        enc.enable_cache(0)
        enc.rerank(["q"], [_docs("a")])
        enc.rerank(["q"], [_docs("a")])
        self.assertEqual(len(enc.forwarded), 2)

    def test_concurrent_requests_are_consistent(self):
        enc = _CountingEncoder()
        enc.enable_cache(1000)
        docs = _docs(*[f"d{i}" for i in range(20)])
        results = []

        def go():
            results.append({d["id"]: s for d, s in enc.rerank(["q"], [docs])[0]})

        threads = [threading.Thread(target=go) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(len(results), 8)
        for r in results[1:]:
            self.assertEqual(r, results[0])


if __name__ == "__main__":
    unittest.main()
