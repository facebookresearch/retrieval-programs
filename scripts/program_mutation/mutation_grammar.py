# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""A typed space of retrieval programs, and the enumeration of mutations within it.

Rather than apply a hand-picked list of edits (scale top_k, add the missing modality, widen the rerank), a
grammar declares what a legal retrieval program is and lets enumeration find the best one: we specify a type
system for pipelines, not a set of pipelines.

Two constraints the design inherits:
1. LAST TURN ONLY. A later turn's query is written after seeing the earlier turn's output, so broadening turn
   1 would change what query the model writes in turn 2 -- replaying turn 2 verbatim against a mutated turn 1
   is counterfactually invalid. Editing only the final retrieval turn leaves no downstream turn to invalidate
   and keeps the whole thing model-free.
2. NO INVENTED QUERIES. Queries are drawn only from those the model already wrote in that turn. RL optimises
   query text; mutation explores the primitive space. Mixing the two would need a model in the loop.
"""

from __future__ import annotations

import ast
import bisect
from dataclasses import dataclass, field

# Breadth and keep ladders.
TOPK_CHOICES: tuple[int, ...] = (3, 5, 10, 20, 50, 100, 200)
KEEP_CHOICES: tuple[int, ...] = (3, 5, 10, 20)
METHOD_CONFIGS: tuple[str, ...] = ("dense", "sparse", "both")
RERANK_SCOPES: tuple[str, ...] = ("call", "union")
MAX_TOPK = 200           # the SDK rejects more, so generating it would only produce a runtime error
DEFAULT_DOC_BUDGET = 20  # documents a program may put in context; a program exceeding it is not runnable


@dataclass(frozen=True)
class Call:
    """One retrieval call. `method` is "dense" or "sparse"."""

    method: str
    query: str
    top_k: int

    def render(self) -> str:
        fn = "dense_retrieval" if self.method == "dense" else "sparse_retrieval"
        return f'{fn}({self.query!r}, top_k={self.top_k})'


@dataclass(frozen=True)
class Program:
    """A last-turn retrieval program: calls, optional fusion, optional rerank, then print.

    `rerank_keep=None` means no rerank -- the raw (optionally fused) list is printed, which is only runnable
    when it fits the document budget.
    """

    calls: tuple[Call, ...]
    fuse: bool
    rerank_query: str | None = None
    rerank_keep: int | None = None
    # `call`: each call's list is reranked against that call's own query and cut to `rerank_keep`, then the cut
    # lists are fused, which is what `mutation_screen.run_program` executes by default. `union`: the fused list
    # is reranked once against `rerank_query`, the form a policy writes as `rerank(q, candidates, top_k=k)`.
    rerank_scope: str = "call"

    # ---- cost, used to order candidates cheapest-first so the selected repair is also the cheapest one ----

    @property
    def n_retrievals(self) -> int:
        """Outgoing retrieval calls. This is the quantity the efficiency argument is about."""
        return len(self.calls)

    @property
    def docs_retrieved(self) -> int:
        return sum(c.top_k for c in self.calls)

    @property
    def docs_printed(self) -> int:
        """Upper bound on documents reaching the context (fusion dedups, so this is a bound, not exact)."""
        if self.rerank_keep is None:
            return self.docs_retrieved
        return self.rerank_keep * (len(self.calls) if self.rerank_scope == "call" else 1)

    @property
    def cost(self) -> tuple[int, int, int]:
        return (self.n_retrievals, self.docs_retrieved, self.docs_printed)

    def render(self, tail: str = "", result_var: str = "") -> str:
        """Python for one `<code>` block, in the idiom the RPL prompts already elicit.

        Style matters: the mutant is spliced into a trace the model is then trained on, so an unfamiliar
        idiom is extra off-policy exposure on top of the edit itself.

        With `tail`/`result_var` from `split_tail`, the model's own post-retrieval lines are kept verbatim
        and only the retrieval is regenerated -- the same locality principle as "no invented queries".
        """
        # A call-scope rerank over several calls is written per call, `fuse(rerank(q1, <call 1>, top_k=k), ...)`,
        # because that is what the gates execute; one rerank of the fused list would print what they never ran.
        per_call = self.rerank_keep is not None and self.rerank_scope == "call" and len(self.calls) > 1
        if per_call:
            inner = "fuse(" + ", ".join(f"rerank({c.query!r}, {c.render()}, top_k={self.rerank_keep})"
                                        for c in self.calls) + ")"
        else:
            inner = f"fuse({', '.join(c.render() for c in self.calls)})" if self.fuse else self.calls[0].render()
        rerank_query = self.calls[0].query if self.rerank_scope == "call" else self.rerank_query
        rerank = (f"rerank({rerank_query!r}, candidates, top_k={self.rerank_keep})"
                  if self.rerank_keep is not None and not per_call else "")
        if tail and result_var:
            head = [f"candidates = {inner}"] if rerank else []
            return "\n".join(head + [f"{result_var} = {rerank or inner}", tail])
        if not rerank:
            iterable = inner
            lines = []
        else:
            # `candidates`, not `docs`: some prompts' only assignment exemplar is `candidates = ...` (others use
            # `docs = ...`). The name is cosmetic to the sandbox but not to the policy -- these lines are
            # spliced into trajectories it is then trained on, so an off-idiom name is gratuitous drift.
            lines = [f"candidates = {inner}"]
            iterable = rerank
        lines += [
            f"for i, doc in enumerate({iterable}):",
            "    print(f\"Doc {i+1}(Title: {doc['title']}) {doc['text']}\")",
        ]
        return "\n".join(lines)


RETRIEVAL_NAMES = frozenset({"dense_retrieval", "sparse_retrieval", "fuse", "rerank"})


def split_tail(code: str) -> tuple[str, str]:
    """Split a `<code>` block into (result variable, the policy's own lines after the last retrieval)."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return "", ""
    last = None
    for node in tree.body:
        if any(isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in RETRIEVAL_NAMES
               for n in ast.walk(node)):
            last = node
    if last is None or last.end_lineno is None:
        return "", ""
    tail = "\n".join(code.splitlines()[last.end_lineno:]).strip("\n")
    if not tail.strip():
        return "", ""
    var = (last.targets[0].id if isinstance(last, ast.Assign) and len(last.targets) == 1
           and isinstance(last.targets[0], ast.Name) else "")
    # A tail with no bound result cannot be reattached -- it would reference a name the new head never binds.
    return (var, tail) if var else ("", "")


def local_ladder(value: int, lo_mult: float = 0.5, hi_mult: float = 1.5, step: int = 5,
                 lo_min: int = 1, hi_max: int = MAX_TOPK) -> tuple[int, ...]:
    """A breadth ladder centered on the value the policy already wrote, rather than the fixed global one."""
    lo = max(lo_min, int(value * lo_mult))
    hi = min(hi_max, int(round(value * hi_mult)))
    rungs = {value} | {x for x in range(step * ((lo + step - 1) // step), hi + 1, step) if x >= lo_min}
    return tuple(sorted(rungs))


def method_config(prog: Program) -> str:
    """`dense` | `sparse` | `both` -- the modality axis, independent of breadth and rerank."""
    methods = {c.method for c in prog.calls}
    return "both" if len(methods) > 1 else next(iter(methods), "none")


def _ladder_steps(a: int, b: int, ladder: tuple[int, ...]) -> int:
    """How many rungs apart two breadths are. Handles a value that is not itself on the ladder."""
    return abs(bisect.bisect_left(ladder, a) - bisect.bisect_left(ladder, b))


def rerank_semantics(prog: Program) -> tuple[str, ...] | None:
    """What the rerank stage scores against, independent of its width: None, `("call",)` or `("union", q)`.

    A single call reranked against its own query is the same program in either scope, so it is `("call",)`.
    """
    if prog.rerank_keep is None:
        return None
    if prog.rerank_scope == "union" and not (len(prog.calls) == 1 and prog.rerank_query == prog.calls[0].query):
        return ("union", prog.rerank_query or "")
    return ("call",)


def _rerank_changed(original: Program, cand: Program) -> int:
    """Does `cand` change what the rerank stage scores against? Width changes are `keep_steps`, not this."""
    def stages(p: Program) -> set:
        s = rerank_semantics(p)
        # A single reranked call has no scope: fusing more lists under its query, either way, keeps the stage.
        return {s, ("union", p.calls[0].query)} if s == ("call",) and len(p.calls) == 1 else {s}
    return int(not stages(original) & stages(cand))


def edit_distance(original: Program, cand: Program,
                  topk_choices: tuple[int, ...] = TOPK_CHOICES,
                  keep_choices: tuple[int, ...] = KEEP_CHOICES) -> tuple[int, ...]:
    """How far `cand` is from the policy's own program, lexicographically, cheapest edit first.

    The components, in priority order -- structure first, magnitude second:
    1. `method_changed` -- dense/sparse/both.
    2. `rerank_changed` -- adding, removing or re-scoping the rerank stage (see `rerank_semantics`).
    3. `queries_changed` -- using a different subset of the turn's own queries.
    4. `topk_steps` -- rungs moved on the breadth ladder.
    5. `keep_steps` -- rungs moved on the rerank-width ladder.

    What remains a genuine judgement rather than an artifact is the priority itself: because the tuple is
    lexicographic and `topk_steps` precedes `keep_steps`, every keep-only edit sorts before any breadth
    edit. That is a claim that re-ranking fewer documents is a smaller change than fetching fewer, which is
    defensible but not obvious. `--repair_order breadth` is the direct test of it.
    """
    o_keep, c_keep = original.rerank_keep, cand.rerank_keep
    o_tk = sorted(c.top_k for c in original.calls) or [0]
    c_tk = sorted(c.top_k for c in cand.calls) or [0]
    pairs = zip(o_tk, c_tk) if len(o_tk) == len(c_tk) else zip([max(o_tk)] * len(c_tk), c_tk)
    return (
        int(method_config(original) != method_config(cand)),
        _rerank_changed(original, cand),
        int({c.query for c in original.calls} != {c.query for c in cand.calls}),
        max((_ladder_steps(a, b, topk_choices) for a, b in pairs), default=0),
        0 if (o_keep is None or c_keep is None) else _ladder_steps(o_keep, c_keep, keep_choices),
    )


def violations(prog: Program, doc_budget: int = DEFAULT_DOC_BUDGET) -> list[str]:
    """Why `prog` is not a legal program, or [] if it is.

    These are the constraints that exclude clearly-bad programs without hand-picking the good ones.
    """
    out = []
    if not prog.calls:
        out.append("no retrieval call")
    if prog.fuse and len(prog.calls) < 2:
        out.append("fuse needs at least two lists")
    if not prog.fuse and len(prog.calls) != 1:
        out.append("more than one call requires fuse to combine them")
    seen = {(c.method, c.query, c.top_k) for c in prog.calls}
    if len(seen) != len(prog.calls):
        out.append("duplicate (method, query, top_k) call")   # a repeated identical call returns the same docs
    for c in prog.calls:
        if not 1 <= c.top_k <= MAX_TOPK:
            out.append(f"top_k {c.top_k} out of range")
        if c.method not in ("dense", "sparse"):
            out.append(f"unknown method {c.method}")
    if (prog.rerank_keep is None) != (prog.rerank_query is None):
        out.append("rerank needs both a query and a keep width")
    if prog.rerank_scope not in RERANK_SCOPES:
        out.append(f"unknown rerank scope {prog.rerank_scope}")
    if prog.rerank_keep is not None and not 1 <= prog.rerank_keep <= MAX_TOPK:
        out.append(f"rerank keep {prog.rerank_keep} out of range")
    if prog.docs_printed > doc_budget:
        out.append(f"prints {prog.docs_printed} documents, over the {doc_budget} budget")
    return out


def is_valid(prog: Program, doc_budget: int = DEFAULT_DOC_BUDGET) -> bool:
    return not violations(prog, doc_budget)


@dataclass
class Space:
    """The enumerable neighborhood for one rollout's last turn.

    `queries` come from that turn's own calls, in first-appearance order, deduplicated.
    """

    queries: tuple[str, ...]
    topk_choices: tuple[int, ...] = TOPK_CHOICES
    keep_choices: tuple[int, ...] = KEEP_CHOICES
    doc_budget: int = DEFAULT_DOC_BUDGET
    method_configs: tuple[str, ...] = METHOD_CONFIGS
    # The query the policy's own union rerank scores against, reused by union candidates so its program stays
    # reachable. None falls back to the first query.
    rerank_query: str | None = None
    _seen: set = field(default_factory=set, repr=False)

    def query_sets(self) -> list[tuple[str, ...]]:
        """Each single query, plus all of them together when there is more than one."""
        singles = [(q,) for q in self.queries]
        return singles + ([tuple(self.queries)] if len(self.queries) > 1 else [])

    def _build(self, qs: tuple[str, ...], method: str, top_k: int, keep: int | None,
               scope: str = "call") -> Program:
        methods = ("dense", "sparse") if method == "both" else (method,)
        calls = tuple(Call(m, q, top_k) for q in qs for m in methods)
        rerank_query = (self.rerank_query or qs[0]) if scope == "union" else qs[0]
        return Program(calls=calls, fuse=len(calls) > 1,
                       rerank_query=rerank_query if keep is not None else None, rerank_keep=keep, rerank_scope=scope)

    @staticmethod
    def _scopes(qs: tuple[str, ...], method: str, keep: int | None) -> tuple[str, ...]:
        # Any multi-call rerank can be written either way. Union is the form RL converges to, and it prints fewer
        # documents, so it wins the cost tie-break when both cover.
        return RERANK_SCOPES if keep is not None and (len(qs) > 1 or method == "both") else ("call",)

    def enumerate(self, original: Program | None = None, prefer: str = "") -> list[Program]:
        """Every legal program in the space, ordered so the first covering hit is the one to train on.

        With `original`, candidates are ordered by `edit_distance` from the policy's own program and only
        then by absolute cost -- the minimal mutation. `prefer` puts candidates that change one axis ahead
        of equally-near ones that do not, because the minimal improving edit is often not the axis whose
        deficit we are trying to fix.
        """
        out: list[Program] = []
        for qs in self.query_sets():
            for method in self.method_configs:
                for top_k in self.topk_choices:
                    for keep in (None, *self.keep_choices):
                        for scope in self._scopes(qs, method, keep):
                            prog = self._build(qs, method, top_k, keep, scope)
                            if not is_valid(prog, self.doc_budget):
                                continue
                            key = prog.render()
                            if key in self._seen:  # e.g. keep >= docs_retrieved renders the same program twice
                                continue
                            self._seen.add(key)
                            out.append(prog)
        if original is None:
            out.sort(key=lambda p: (p.cost, p.render()))
        elif prefer == "breadth":
            base = {c.top_k for c in original.calls}
            out.sort(key=lambda p: (0 if {c.top_k for c in p.calls} != base else 1,
                                    edit_distance(original, p, self.topk_choices, self.keep_choices),
                                    p.cost, p.render()))
        elif prefer == "rerank":
            had = original.rerank_keep is not None
            out.sort(key=lambda p: (0 if (p.rerank_keep is not None) != had else 1,
                                    edit_distance(original, p, self.topk_choices, self.keep_choices),
                                    p.cost, p.render()))
        else:
            out.sort(key=lambda p: (edit_distance(original, p, self.topk_choices, self.keep_choices),
                                    p.cost, p.render()))
        return out


def space_for(calls: list[dict], ladder: str = "global", keep: int | None = None,
              ladder_lo: float = 0.5, rerank_query: str | None = None, **kw) -> Space:
    """Build the space from a rollout's parsed last-turn calls (`mutation_screen.parse_calls` output)."""
    queries: list[str] = []
    for c in calls:
        if c["query"] not in queries:
            queries.append(c["query"])
    if ladder == "relative":
        emitted = max((c["top_k"] for c in calls), default=20)
        kw.setdefault("topk_choices", local_ladder(emitted, lo_mult=ladder_lo))
        kw.setdefault("keep_choices", local_ladder(keep if keep else emitted, lo_mult=ladder_lo))
    elif ladder != "global":
        raise ValueError(f"ladder must be 'global' or 'relative', got {ladder!r}")
    return Space(queries=tuple(queries), rerank_query=rerank_query, **kw)


def renders_to_valid_python(prog: Program) -> bool:
    """A program that does not parse would fail inside the sandbox as an opaque error, not a bad retrieval."""
    try:
        ast.parse(prog.render())
        return True
    except SyntaxError:
        return False
