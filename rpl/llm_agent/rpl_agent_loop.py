# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Multi-turn rollout with AgentLoop.

Ports the per-trajectory logic of `generation.py::run_llm_loop` onto `AgentLoopBase`.
`response_mask` is 1 for model-generated tokens (trainable) and 0 for injected observations
(`<information>` / `<output>`, loss-masked).

Registered by `rpl/config/agent_loop.yaml` (referenced via
`actor_rollout_ref.rollout.agent.agent_loop_config_path`) and selected with
`actor_rollout_ref.rollout.agent.default_agent_loop=rpl`.
"""

from __future__ import annotations

import json
import re
import requests
import time
from typing import Any, Optional
import uuid

from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopOutput, register
from verl.utils.rollout_trace import rollout_trace_op
from verl.workers.rollout.replica import TokenOutput

from .code_executor import CodeExecutor, ExecResult


_ACTION_RE = re.compile(r"<(search|code|answer)>(.*?)</\1>", re.DOTALL)
_LEADING_THINK_CLOSE = re.compile(r"\A(?:\s*</think>)+\s*")


def _strip_leading_think_close(text: str) -> str:
    """Drop leading orphan `</think>` tags so `_ACTION_RE` sees the real action.

    Only ever removes tags BEFORE any content -- a `</think>` that closes a genuine `<think>` block is
    preceded by that block and is left alone.
    """
    return _LEADING_THINK_CLOSE.sub("", text, count=1) if text else text


def _is_degenerate_repeat(text: str, frag: int = 24, times: int = 10) -> bool:
    """True when a short fragment repeats `times`+ times -- a generation stuck in a loop.

    Mode-agnostic on purpose: it catches any degenerate repeat, not just the `</think>` one, and healthy
    generations do not repeat a 24-char fragment ten times. Checked against the tail so a long, legitimate
    answer that merely ends in a loop is still caught.
    """
    if not text or len(text) < frag * times:
        return False
    tail = text[-6000:]
    for i in range(0, min(len(tail) - frag, 2000), 200):
        if tail.count(tail[i:i + frag]) >= times:
            return True
    return False


# Invalid-action retry messages adapted from generation.py::execute_predictions.
_INVALID_SAT = (
    "\n<think>My previous action is invalid. If I want to search, I should put the query between "
    "<search> and </search>. If I want to give the final answer, I should put the answer between "
    "<answer> and </answer>. Let me try again.</think>\n"
)
_INVALID_SAC = (
    "\n<think>My previous action is invalid. If I want to search, I should write Python between "
    "<code> and </code> that uses the search SDK and prints the results. If I want to give the final "
    "answer, I should put it between <answer> and </answer>. Let me try again.</think>\n"
)
_INVALID_MARAG = (
    "\n<think>My previous action is invalid. I must call exactly ONE tool per turn, writing Python between "
    "<code> and </code> that calls one of DenseSearch, KeywordSearch or DocumentFilter, or that only "
    "combines documents I already gathered. If I want to give the final answer, I should put it between "
    "<answer> and </answer>. Let me try again.</think>\n"
)
# Tool calls counted for the one-per-turn constraint. Aggregation is plain Python and is NOT a tool, so a
# block that only counts or sorts already-gathered documents is always legal.
_MARAG_TOOL_RE = re.compile(r"\b(?:DenseSearch|KeywordSearch|DocumentFilter)\s*\(")

# Injected when the model tries to answer before making `min_searches` retrieval calls: replaces the
# answer with a "search again" nudge so the loop continues. <think>-wrapped to stay well-formed.
_PREMATURE_ANSWER = (
    "\n<think>I don't have enough evidence to answer yet. Let me search again before giving the final "
    "answer.</think>\n"
)
# Used by `condense_docs` (see AgentLoop.__init__). Lives here with the other RUNTIME-injected strings rather
# than in scripts/data_process/prompts.py: those prompts are baked into the parquets at build time, this one
# is applied per retrieval during the rollout, so it must be importable by the loop.
_CONDENSE_INSTR = (
    "You are given a search query and the documents retrieved for it. Extract only the information from "
    "those documents that helps answer the query, as a short factual summary. Do not add anything the "
    "documents do not say. If none of them are relevant, reply exactly: {empty}\n\n"
    "Query: {query}\n\nRetrieved documents:\n{docs}\n\nExtracted information:"
)
_CONDENSE_EMPTY = "No helpful information found."


def _sp_print(show_doc_id: bool, doc_chars: int = 0) -> str:
    """The print line the SaT+ snippet uses, optionally truncating each document to `doc_chars`.

    Built rather than templated because the cut has to land inside the f-string's replacement fields
    (`{doc['text'][:600]}`), which a `.format()` placeholder cannot express without brace escaping -- and
    escaping here previously emitted the literal text `{i+1}` into observations.
    """
    cut = f"[:{doc_chars}]" if doc_chars > 0 else ""
    ids = "id: {doc['id']}, " if show_doc_id else ""
    return 'print(f"Doc {i+1}(' + ids + "Title: {doc['title']" + cut + "}) {doc['text']" + cut + '}")'


# Single braces below are substituted into the template by `.format()`, not part of it.
_SEARCH_PLUS_SNIPPETS = {
    "fused": """\
candidates = dedup(dense_retrieval({q}, top_k={k}) + sparse_retrieval({q}, top_k={k}))
for i, doc in enumerate(rerank({q}, candidates, top_k={t})):
    {p}
""",
    "dense": """\
docs = dense_retrieval({q}, top_k={k})
for i, doc in enumerate(rerank({q}, docs, top_k={t})):
    {p}
""",
}


def _passages2string(retrieval_result, show_doc_id: bool = False, doc_chars: int = 0) -> str:
    """Format retriever docs into the `<information>` body.

    `show_doc_id` additionally prints each doc's corpus id -- needed for id-answer tasks (GlobalQA).
    """
    out = ""
    for idx, doc_item in enumerate(retrieval_result):
        doc = doc_item["document"]
        content = doc["contents"]
        title = content.split("\n")[0]
        text = "\n".join(content.split("\n")[1:])
        # Truncate per document, not the concatenation.
        if doc_chars > 0:
            title = title[:doc_chars]
            text = text[:doc_chars]
        if show_doc_id:
            out += f"Doc {idx + 1}(id: {doc.get('id')}, Title: {title}) {text}\n"
        else:
            out += f"Doc {idx + 1}(Title: {title}) {text}\n"
    return out


@register("rpl")
class RPLAgentLoop(AgentLoopBase):
    """Multi-turn search rollout loop.

    Usage: set `actor_rollout_ref.rollout.agent.default_agent_loop=rpl` and point
    `agent_loop_config_path` at `rpl/config/agent_loop.yaml`.
    """

    def __init__(self, *args, **kwargs):
        # Search-R1 uses a base model with raw-text prompts (no chat template), but AgentLoopBase.__init__
        # calls initialize_system_prompt -> tokenizer.apply_chat_template, which errors without a template.
        # Install a raw-passthrough template (concatenate message contents, no special tokens) if none is
        # set, so init succeeds. Our rollout tokenizes raw text (_prompt_text/_tokenize) regardless, so
        # this is parity-neutral -- the resulting (empty) system prompt is never used by this loop.
        tok = kwargs.get("tokenizer")
        if tok is not None and getattr(tok, "chat_template", None) is None:
            tok.chat_template = "{% for message in messages %}{{ message['content'] }}{% endfor %}"
        super().__init__(*args, **kwargs)
        self.response_length = self.rollout_config.response_length
        # Custom config block (see module docstring). `search_url` is required. Lives at the config
        # top level (`config.rpl`), NOT under `rollout.agent` -- that is a strict `AgentLoopConfig`
        # dataclass that rejects unknown keys; the top level tolerates extra keys (like `distillation`).
        cfg = self.config.rpl
        self.mode = cfg.get("mode", "search")  # "search", "code", "marag", "search_plus"
        self.search_url = cfg["search_url"]
        self.rerank_url = cfg.get("rerank_url") or "http://127.0.0.1:6980/rerank"
        # Optional published-endpoint JSON so a restarted reranker is hot-swapped without relaunching (CodeExecutor).
        self.rerank_endpoint_file = cfg.get("rerank_endpoint_file") or ""
        # Same for the retriever: re-resolve the published endpoint so training recovers if a restarted standalone
        # retriever changes address (no relaunch, no mid-run hang). Empty -> static search_url.
        self.retriever_endpoint_file = cfg.get("retriever_endpoint_file") or ""
        self._su = [self.search_url, 0.0]  # [resolved url, ts] cache for the retriever hot-swap
        self._su_ttl = 30.0
        self.topk = cfg.get("topk", 3)
        self.max_turns = cfg.get("max_turns", 4)
        self._invalid = _INVALID_SAC if self.mode == "code" else _INVALID_SAT
        if self.mode == "marag":
            self._invalid = _INVALID_MARAG
        self._code_executor: Optional[CodeExecutor] = None
        if self.mode in ("code", "search_plus", "marag"):
            self._code_executor = CodeExecutor(
                retriever_url=self.search_url,
                rerank_url=self.rerank_url,
                rerank_endpoint_file=self.rerank_endpoint_file,
                retriever_endpoint_file=self.retriever_endpoint_file,
                topk_default=self.topk,
                max_search_calls=cfg.get("sandbox_max_search_calls", 0),
                rerank_max_docs=int(cfg.get("rerank_max_docs", 0)),
                max_state_bytes=int(cfg.get("max_state_bytes", 4 * 1024 * 1024)),
                rerank_relevance_threshold=float(cfg.get("rerank_relevance_threshold", 0.5)),
                marag_filter_frac=float(cfg.get("marag_filter_frac", 0.0)),
                marag_topk=int(cfg.get("marag_topk", 20)),
                timeout_sec=cfg.get("sandbox_timeout_sec", 120.0),
            )
        # Stop generation at the first close tag so we can act on the turn.
        self._stop = ["</answer>", "</code>" if self.mode in ("code", "marag") else "</search>"]
        # Cap per-turn generation (max_response_length) and truncate each observation to a fixed token budget
        # (max_obs_length). 0 = unbounded (general default).
        self.max_gen_tokens = int(cfg.get("max_gen_tokens", 0))
        self.max_obs_length = int(cfg.get("max_obs_length", 0))
        # Per-document character cap for the text observations; 0 = off (upstream behaviour).
        self.doc_chars = int(cfg.get("doc_chars", 0))
        # SaT+ (`mode=search_plus`): the model still emits <search>[query]</search> and still reads a plain
        # <information> block, but the BACKEND answers that one query with the full pipeline --
        # dense + sparse -> fuse -> rerank.
        self.search_plus_candidates = int(cfg.get("search_plus_candidates", 20))
        self.search_plus_topk = int(cfg.get("search_plus_topk", 5))
        # which fixed program runs behind <search>: "fused" (dense+sparse->fuse->rerank) or
        # "dense" (dense->rerank). Unknown values fail here rather than silently picking one.
        self.search_plus_pipeline = cfg.get("search_plus_pipeline", "fused")
        if self.mode == "search_plus" and self.search_plus_pipeline not in _SEARCH_PLUS_SNIPPETS:
            raise ValueError(f"unknown search_plus_pipeline={self.search_plus_pipeline!r}; expected one of {sorted(_SEARCH_PLUS_SNIPPETS)}")
        # Prompt formatting: "raw" (default) concatenates the raw_prompt contents and tokenizes with NO chat
        # template. "apply" uses the model's own chat_template (add_generation_prompt), for instruct checkpoints
        # trained with one. In "raw" mode the model's chat_template is irrelevant to the rollout.
        self.chat_template_mode = cfg.get("chat_template_mode", "raw")
        # Only for chat_template_mode="apply": forwarded to apply_chat_template as enable_thinking. True
        # (default) keeps an instruct model's native reasoning (e.g. Qwen3 <think>...</think>).
        self.enable_thinking = bool(cfg.get("enable_thinking", True))
        # Only for chat_template_mode="raw": add BOS to the INITIAL prompt. Default True.
        self.add_bos_to_prompt = bool(cfg.get("add_bos_to_prompt", True))
        # Require >= this many retrieval calls before an <answer> is accepted (0 = off). A premature answer
        # is replaced with a "search again" nudge and the loop continues.
        self.min_searches = int(cfg.get("min_searches", 0))
        # Carry the sandbox namespace between <code> blocks of ONE rollout fresh-namespace behaviour. Off,
        # `docs = retrieve(...)` in one turn and `rerank(q, docs, 3)` in the next is a guaranteed NameError.
        self.persist_sandbox_state = bool(cfg.get("persist_sandbox_state", False))
        # For id-answer tasks (GlobalQA): print the corpus id in each <information> doc so SaT can answer
        # id-based aggregations (Count/Sort/MinMax). Default off -> verbatim wiki-18 <information> format.
        self.show_doc_id = bool(cfg.get("show_doc_id", False))
        # Condense each retrieval's passages with the policy model before injecting them, instead of pasting
        # the raw top-k. The Search-o1 baseline is exactly this switched on.
        self.condense_docs = bool(cfg.get("condense_docs", False))
        self.condense_max_tokens = int(cfg.get("condense_max_tokens", 256))
        # Training-free baselines (direct / CoT / RAG) are ONE forward pass, not an agent rollout: take the
        # first generation verbatim, with no observation step and no forced re-answer.
        self.single_pass = bool(cfg.get("single_pass", False))

    def _tokenize(self, text: str, add_special: bool = False) -> list[int]:
        # Raw-text tokenize, no chat template. add_special=True adds BOS. Observations are mid-sequence,
        # so they must stay BOS-free (add_special=False).
        return self.tokenizer(text, add_special_tokens=add_special)["input_ids"]

    def _cap_observation(self, obs: str, obs_ids: list[int]) -> list[int]:
        """`obs_ids` cut to `max_obs_length` tokens, keeping the closing tag."""
        if self.max_obs_length <= 0 or len(obs_ids) <= self.max_obs_length:
            return obs_ids
        # Reserve room for the closing tag. A bare slice can cut it off, and the model then reads an
        # UNTERMINATED observation block -- malformed input it has to guess its way out of, and any
        # `<output>...</output>` parser downstream merges that block with everything after it.
        tail = "</information>\n\n" if obs.rstrip().endswith("</information>") else "</output>\n\n"
        tail_ids = self._tokenize(tail)
        return obs_ids[: max(0, self.max_obs_length - len(tail_ids))] + tail_ids

    def _prefix_ids(self, prefix: str) -> list[int]:
        """Token ids of a `response_prefix`, its final observation capped like a live one.

        The program-mutation verify and control prefixes end with an observation built offline by re-running a
        program. Nothing capped it, so 29-42% of them ran past 2048 tokens: longer evidence than the policy ever
        read in training. Earlier observations come from the rollout's own text and were capped when it ran.
        """
        prefix_ids = self._tokenize(prefix)
        if self.max_obs_length <= 0 or not prefix.rstrip().endswith(("</output>", "</information>")):
            return prefix_ids
        start = max(prefix.rfind("\n\n<output>"), prefix.rfind("\n\n<information>"))
        if start < 0:
            return prefix_ids
        obs = prefix[start:]
        obs_ids = self._tokenize(obs)
        if len(obs_ids) <= self.max_obs_length:
            return prefix_ids
        return self._tokenize(prefix[:start]) + self._cap_observation(obs, obs_ids)

    def _current_search_url(self) -> str:
        """Re-resolve the retriever URL from its published endpoint JSON (TTL) so it recovers if a restarted
        standalone retriever changes address. Empty endpoint file -> static search_url."""
        if not self.retriever_endpoint_file:
            return self.search_url
        now = time.time()
        if now - self._su[1] >= self._su_ttl:
            self._su[1] = now
            try:
                with open(self.retriever_endpoint_file) as f:
                    u = json.load(f).get("url")
                if u:
                    self._su[0] = u
            except (OSError, ValueError):
                pass
        return self._su[0]

    def _encode_prompt(self, raw_prompt: Any, enable_thinking: Optional[bool] = None) -> list[int]:
        """Tokenize a prompt the way this loop is configured to: chat template (`apply`) or raw text."""
        if enable_thinking is None:
            enable_thinking = self.enable_thinking
        if self.chat_template_mode == "apply":
            # return_dict=True + ["input_ids"]: apply_chat_template returns a BatchEncoding dict in this
            # transformers version, so a bare list(...) would yield the KEYS ('input_ids', ...) and feed
            # vLLM the string "input_ids" as a token. Take the ids explicitly.
            enc = self.tokenizer.apply_chat_template(
                raw_prompt,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                enable_thinking=enable_thinking,
            )
            return list(enc["input_ids"])
        return self._tokenize(_prompt_text(raw_prompt), add_special=self.add_bos_to_prompt)

    async def _condense(self, query: str, docs: str) -> str:
        """Distill `docs` down to what answers `query`, using the policy model itself.

        An empty completion becomes the explicit "no helpful information" string rather than falling back to
        the raw passages. A silent fallback would turn a broken condenser into "condensing does not help",
        which is indistinguishable from a real result.
        """
        prompt = [{"role": "user", "content": _CONDENSE_INSTR.format(
            empty=_CONDENSE_EMPTY, query=query, docs=docs)}]
        out: TokenOutput = await self.server_manager.generate(
            request_id=uuid.uuid4().hex,
            prompt_ids=self._encode_prompt(prompt, enable_thinking=False),
            sampling_params={"temperature": 0.0, "max_tokens": self.condense_max_tokens},
        )
        return self.tokenizer.decode(out.token_ids).strip() or _CONDENSE_EMPTY

    def _retrieve(self, query: str) -> tuple[str, Optional[ExecResult], list]:
        """Documents for one `<search>`: the observation text, the sandbox result, and the doc ids."""
        if self.mode != "search_plus":
            payload = {"queries": [query], "topk": self.topk, "return_scores": True}
            result = requests.post(self._current_search_url(), json=payload).json()["result"][0]
            # Read the id exactly as `_passages2string` and the SDK do (`item["document"]["id"]`, no
            # coercion): `_set_f1` intersects raw sets, so casting here would silently miss every gold id.
            ids = [d.get("id") for d in (item.get("document", item) for item in result)]
            return (_passages2string(result, self.show_doc_id, self.doc_chars), None,
                    [[i for i in ids if i is not None]])
        # SaT+: the same single text query, answered by the whole code pipeline behind the interface.
        code = _SEARCH_PLUS_SNIPPETS[self.search_plus_pipeline].format(
            q=json.dumps(query), k=self.search_plus_candidates,
            t=self.search_plus_topk, p=_sp_print(bool(self.show_doc_id), self.doc_chars)
        )
        r = self._code_executor.run_one(code)
        if r.error or r.timed_out:
            reason = r.error or f"timed out after {self._code_executor.timeout_sec}s"
            raise RuntimeError(f"search_plus pipeline failed: {reason}")
        return r.stdout, r, r.call_doc_ids

    def _run_code(self, code: str, state: dict) -> ExecResult:
        return self._code_executor.run_batch_detailed([code], [state])[0]

    async def _observe(self, text: str, code_state: Optional[dict] = None
                       ) -> tuple[Optional[str], int, bool, int, int, list, list, int, list]:
        """Map a turn to (obs | None, n_search, done, n_dense, n_sparse, call_docs, rerank_call_docs, n_rerank,
        call_doc_ids).

        Mirrors execute_predictions per-action: answer -> done; search -> <information>; code ->
        <output>; anything else -> the invalid-retry message. None observation means the trajectory
        is finished (answered). The dense/sparse split and per-call retrieved/reranked doc texts (for
        answer recall/precision + rerank-precision scoring in the reward layer) are nonzero for RPL and for
        SaT+ -- both reach the retrievers through the sandbox -- and zero for plain SaT, whose `<search>` is
        one retriever POST already counted by `search_calls`.

        `code_state` (when `persist_sandbox_state` is on) is this rollout's sandbox namespace, mutated in
        place so the next `<code>` block can use variables the previous one defined. It must be caller-owned
        rather than stored on `self`: one AgentLoop instance serves many rollouts, so state on `self` would
        leak one rollout's variables into another's namespace.

        `n_rerank` is threaded separately from `rerank_call_docs` rather than inferred from its length: the
        sandbox increments its counter BEFORE the HTTP post but only records docs on success, so a rerank
        that times out is a real outgoing API call with no docs. The efficiency reward must charge for it.
        """
        # An orphan leading `</think>` otherwise hides a perfectly good action from `_ACTION_RE`.
        m = _ACTION_RE.search(_strip_leading_think_close(text))
        if not m:
            # A generation stuck repeating a fragment will never produce an action, and re-prompting it
            # only burns another turn -- end the trajectory instead of spending the budget on a dead loop.
            if _is_degenerate_repeat(text):
                return None, 0, True, 0, 0, [], [], 0, []
            return self._invalid, 0, False, 0, 0, [], [], 0, []
        action, content = m.group(1), m.group(2).strip()
        if action == "answer":
            return None, 0, True, 0, 0, [], [], 0, []
        if action == "search":
            docs, r, doc_ids = await self.loop.run_in_executor(None, self._retrieve, content)
            if self.condense_docs:
                docs = await self._condense(content, docs)
            obs = f"\n\n<information>{docs.strip()}</information>\n\n"
            if r is None:
                # Plain SaT: the call counters stay zero because its one POST is already in `search_calls`,
                # but the doc IDs are real and `doc_f1@20` is computed from them.
                return obs, 1, False, 0, 0, [], [], 0, doc_ids
            return (obs, 1, False, r.dense_calls, r.sparse_calls, r.call_docs, r.rerank_call_docs,
                    r.rerank_calls, doc_ids)
        if action == "code":
            if self._code_executor is None:
                return self._invalid, 0, False, 0, 0, [], [], 0, []
            # MARAG-R1 allows one tool per turn.
            if self.mode == "marag" and len(_MARAG_TOOL_RE.findall(content)) > 1:
                return self._invalid, 0, False, 0, 0, [], [], 0, []
            seed = dict(code_state) if (self.persist_sandbox_state and code_state) else {}
            r: ExecResult = await self.loop.run_in_executor(None, self._run_code, content, seed)
            if self.persist_sandbox_state and code_state is not None:
                code_state.clear()
                code_state.update(r.state)
            # search_calls = the retrievals the sandbox ACTUALLY made (ground-truth reward gate); dense/sparse split it.
            return (f"\n\n<output>{r.observation().strip()}</output>\n\n", r.search_calls, False,
                    r.dense_calls, r.sparse_calls, r.call_docs, r.rerank_call_docs, r.rerank_calls,
                    r.call_doc_ids)
        return self._invalid, 0, False, 0, 0, [], [], 0, []

    @rollout_trace_op
    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        # Validation rolls out greedily (temperature 0, the standard for deterministic EM eval); training
        # samples (temperature > 0). v0.8.0's streaming reward path drops batch meta_info, so we thread the
        # train/val distinction to the reward manager via extra_fields, inferring it from the sampling
        # temperature here.
        is_validation = float(sampling_params.get("temperature", 1.0)) == 0.0
        prompt_ids = self._encode_prompt(kwargs["raw_prompt"])
        sampling_params = dict(sampling_params)
        sampling_params["stop"] = self._stop
        sampling_params["include_stop_str_in_output"] = True  # keep the close tag in the response
        if self.max_gen_tokens > 0:
            sampling_params["max_tokens"] = self.max_gen_tokens  # per-turn generation cap (v0.1 parity)

        response_ids: list[int] = []
        response_mask: list[int] = []  # 1 = generated (trainable), 0 = injected observation
        search_calls = 0
        dense_calls = 0
        sparse_calls = 0
        retrieved_docs: list = []      # per retrieval call: [doc_contents,...] (answer recall/precision, reward layer)
        reranked_docs: list = []       # per rerank() call: [doc_contents,...] (rerank precision diagnostic)
        retrieved_doc_ids: list = []   # per retrieval call: [corpus_id,...] (GlobalQA document-coverage F1)
        rerank_calls = 0               # rerank() invocations INCLUDING failed ones (efficiency-reward cost)
        num_turns = 0
        done = False
        code_state: dict = {}          # this rollout's sandbox namespace; local, so rollouts cannot cross-talk

        # Counterfactual continuation (program-mutation the verify and control stages): a row carrying `response_prefix`
        # resumes from that text instead of starting empty. The prefix is the policy's own earlier turns with one
        # retrieval program swapped out, so the continuation measures whether repaired evidence changes the answer.
        # Masked 0 throughout: the prefix is given, not generated by this episode, so it contributes no gradient if
        # ever run under training. `num_turns` counts only this episode's turns; the prefix's turns are not re-counted.
        prefix = (kwargs.get("extra_info") or {}).get("response_prefix")
        if prefix:
            prefix_ids = self._prefix_ids(prefix)
            if len(prefix_ids) >= self.response_length:
                # Silently truncating here would yield an empty continuation scored as a wrong answer.
                raise ValueError(f"response_prefix is {len(prefix_ids)} tokens but response_length is "
                                 f"{self.response_length}, leaving no room to generate; raise "
                                 f"data.max_response_length (MAX_RESPONSE_LEN).")
            response_ids += prefix_ids
            response_mask += [0] * len(prefix_ids)

        async def _generate() -> list[int]:
            out: TokenOutput = await self.server_manager.generate(
                request_id=uuid.uuid4().hex,
                prompt_ids=prompt_ids + response_ids,
                sampling_params=sampling_params,
            )
            return out.token_ids

        # Run up to max_turns of (generate -> observe). Break on a final <answer> or length cap.
        for _ in range(self.max_turns):
            gen = await _generate()
            response_ids += gen
            response_mask += [1] * len(gen)
            num_turns += 1
            if len(response_ids) >= self.response_length:
                break
            if self.single_pass:
                done = True
                break
            obs, n_sc, done, n_dc, n_spc, cd, rcd, n_rr, cdi = await self._observe(self.tokenizer.decode(gen), code_state)
            rerank_calls += n_rr
            search_calls += n_sc
            dense_calls += n_dc
            sparse_calls += n_spc
            retrieved_docs += cd
            retrieved_doc_ids += cdi
            reranked_docs += rcd
            if done and search_calls < self.min_searches:
                # Answered too early -- require more retrieval before accepting the answer.
                done, obs = False, _PREMATURE_ANSWER
            if done:
                break
            obs_ids = self._cap_observation(obs, self._tokenize(obs))
            response_ids += obs_ids
            response_mask += [0] * len(obs_ids)
            num_turns += 1
            if len(response_ids) >= self.response_length:
                break

        # If the model never answered within max_turns, force one final answer generation.
        if not done and len(response_ids) < self.response_length:
            gen = await _generate()
            response_ids += gen
            response_mask += [1] * len(gen)
            num_turns += 1

        return AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=response_ids[: self.response_length],
            response_mask=response_mask[: self.response_length],
            num_turns=num_turns,
            search_calls=search_calls,
            metrics={},
            extra_fields={
                "code_search_calls": search_calls,
                "code_dense_calls": dense_calls,
                "code_sparse_calls": sparse_calls,
                "is_validation": is_validation,
                "retrieved_docs": retrieved_docs,
                "reranked_docs": reranked_docs,
                "retrieved_doc_ids": retrieved_doc_ids,
                "code_rerank_calls": rerank_calls,
            },
        )


def _prompt_text(raw_prompt: Any) -> str:
    """Extract the raw prompt text."""
    if isinstance(raw_prompt, str):
        return raw_prompt
    if isinstance(raw_prompt, (list, tuple)):
        return "".join(m["content"] if isinstance(m, dict) and "content" in m else str(m) for m in raw_prompt)
    return str(raw_prompt)
