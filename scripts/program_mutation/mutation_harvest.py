# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Harvest failed rollouts and find the cheapest grammar repair for each.

Place in the pipeline: RL -> harvest -> verify (re-answer) -> control (resample control) -> SFT -> resume RL.
This module is the model-free part: it decides which rows are repairable and what the cheapest repair is, using
retrieval only. The verify and control stages need generation and are separate.

Usage (normally via `scripts/run_mutation_round.sh`; needs a retriever and a reranker serving):
    python scripts/program_mutation/mutation_harvest.py --dump <round>/rollouts/window.jsonl \
        --url http://127.0.0.1:8000/retrieve --rerank_url http://127.0.0.1:6980/rerank --out repairs.jsonl
"""

from __future__ import annotations

import argparse
import ast
import concurrent.futures
import json
import re
import sys

from mutation_grammar import Call, Program, rerank_semantics, space_for, split_tail
from mutation_screen import (CODE, answer_in, doc_f1, golden_doc_ids, golds, has_var_query, load_supporting,
                             parse_calls, qkey, run_program, supporting_covered)

RETRIEVAL_FNS = ("dense_retrieval", "sparse_retrieval", "search")


class Unrepresentable(ValueError):
    """The block's rerank stage is not a program the grammar can express, so it cannot be replayed faithfully."""


def last_turn(output: str) -> str:
    """The final `<code>` block that issues a retrieval, or "" if there is none.

    The prompt itself contains exemplar calls and the prose mentions `<code>`/`</code>`, so blocks are taken
    from the model's output only and a block with no retrieval call is skipped rather than treated as empty.
    """
    for block in reversed(CODE.findall(output or "")):
        if parse_calls(f"<code>{block}</code>"):
            return block
    return ""


def _call_name(node) -> str:
    return node.func.id if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) else ""


def _arg(node: ast.Call, pos: int, name: str):
    return node.args[pos] if len(node.args) > pos else next((k.value for k in node.keywords if k.arg == name), None)


def _literal(node):
    return node.value if isinstance(node, ast.Constant) else None


def rerank_stage(block: str, calls: tuple[Call, ...]) -> tuple[str, str | None, int | None]:
    """`(scope, query, keep)` of the block's rerank, read from its syntax tree; `("call", None, None)` without one.

    `call` is one rerank per retrieval call against that call's own query; `union` is one rerank of the fused list
    against one query. Anything else (a threshold rerank without a literal `top_k`, mixed widths, a rerank of only
    some calls) raises `Unrepresentable` rather than replaying a different program.
    """
    if "rerank(" not in block:
        return "call", None, None
    try:
        tree = ast.parse(block)
    except SyntaxError as e:
        raise Unrepresentable("the block does not parse") from e
    stages = []
    for node in ast.walk(tree):
        if _call_name(node) != "rerank":
            continue
        query, docs, keep = _literal(_arg(node, 0, "query")), _arg(node, 1, "docs"), _literal(_arg(node, 2, "top_k"))
        if not isinstance(query, str):
            raise Unrepresentable("rerank query is not a string literal")
        if not isinstance(keep, int) or isinstance(keep, bool):
            raise Unrepresentable("rerank without a literal top_k keeps a model-judged number of documents")
        wrapped = _literal(_arg(docs, 0, "query")) if _call_name(docs) in RETRIEVAL_FNS else None
        stages.append((query, wrapped, keep))
    if not stages:
        return "call", None, None
    if len({keep for _, _, keep in stages}) > 1:
        raise Unrepresentable("reranks with different widths")
    keep = stages[0][2]
    if (len(stages) == len(calls) and all(q == w for q, w, _ in stages)
            and sorted(w for _, w, _ in stages) == sorted(c.query for c in calls)):
        return "call", calls[0].query, keep
    if len(stages) > 1:
        raise Unrepresentable("several reranks that are not one per retrieval call")
    query, wrapped, _ = stages[0]
    if len(calls) > 1 and wrapped is not None:
        raise Unrepresentable("rerank applies to one of several retrieval calls")
    return ("call" if len(calls) == 1 and query == calls[0].query else "union"), query, keep


def parse_program(block: str) -> Program | None:
    """Reconstruct the block's program: retrieval calls, whether they are fused, and any rerank width.

    Returns None when the block cannot be represented (no literal-query call, or an `Unrepresentable` rerank),
    which is the same condition that makes it unreplayable.
    """
    calls = parse_calls(f"<code>{block}</code>")
    if not calls:
        return None
    typed = tuple(Call(c["method"] or "dense", c["query"], c["top_k"]) for c in calls)
    try:
        scope, rq, keep = rerank_stage(block, typed)
    except Unrepresentable:
        return None
    return Program(calls=typed, fuse=len(typed) > 1, rerank_query=rq, rerank_keep=keep, rerank_scope=scope)


def covered(docs, row: dict, gold_mode: str, supporting: dict) -> bool:
    """Did this document set deliver the answerable condition for `row`?"""
    if gold_mode == "answer_string":
        return answer_in(docs, golds(row.get("gts")))
    sup = supporting[qkey(row.get("input"))]
    return supporting_covered(docs, sup) == len(sup)


def score_docs(docs, row: dict) -> float:
    """Doc-F1 of a retrieved set against the row's `golden_doc_ids` (GlobalQA)."""
    return doc_f1(docs, golden_doc_ids(row.get("gts")))


def execute(prog: Program, url: str, rerank_url: str, cache: dict) -> dict:
    """Run a Program through the retriever/reranker, caching by (calls, keep, rerank semantics)."""
    calls = [{"method": c.method, "query": c.query, "top_k": c.top_k} for c in prog.calls]
    key = json.dumps([calls, prog.rerank_keep, rerank_semantics(prog)], sort_keys=True)
    if key not in cache:
        cache[key] = run_program(url, rerank_url, calls, prog.rerank_keep, prog.rerank_scope, prog.rerank_query)
    return cache[key]


def select_repair(row: dict, a, supporting: dict) -> tuple[str, dict | None]:
    """`(status, record)` for one failed rollout.

    Status is one of:
      * `already_covered` -- the row's own program did get the gold into context, so its failure is a
        conversion failure, not a retrieval one.
      * `no_repair` -- a genuine retrieval miss that no program in the space fixes. The evidence is likely
        absent from the corpus, or absent from the neighborhood the observed queries can reach.
      * `repaired` -- the cheapest covering program, with its cost.
      * `unparseable` / `unrepresentable` -- no literal-query retrieval call, or a rerank stage the grammar cannot
        express (see `rerank_stage`); either way the program cannot be replayed faithfully.

    Candidates are enumerated best-first, so the first hit is the selected repair and the scan stops --
    which is what keeps this affordable at a few hundred candidates per row. `--repair_order nearest` (the
    default) orders by `edit_distance` from the policy's own program, so the repair is the minimal mutation.
    """
    block = last_turn(row.get("output"))
    original = parse_program(block)
    if original is None:
        return ("unrepresentable" if parse_calls(f"<code>{block}</code>") else "unparseable"), None
    # Keep whatever the policy does with the retrieved documents; the mutation owns retrieval only.
    result_var, tail = split_tail(block)
    cache: dict = {}
    base_docs = execute(original, a.url, a.rerank_url, cache)
    if a.gold_mode == "doc_f1":
        # No "already_covered" branch: doc-F1 is comparative, so every row is a candidate for improvement
        # and the baseline is the original program's own score.
        base = score_docs(base_docs, row)
        if not golden_doc_ids(row.get("gts")):
            return "no_gold_docs", None
    elif covered(base_docs, row, a.gold_mode, supporting):
        return "already_covered", None
    space = space_for([{"method": c.method, "query": c.query, "top_k": c.top_k} for c in original.calls],
                      ladder=a.ladder, keep=original.rerank_keep, ladder_lo=a.ladder_lo,
                      rerank_query=original.rerank_query if original.rerank_scope == "union" else None,
                      doc_budget=a.doc_budget)
    first, all_covering = None, []
    ordered = space.enumerate(original if a.repair_order != "cheapest" else None,
                              prefer=a.repair_order if a.repair_order in ("breadth", "rerank") else "")
    for i, cand in enumerate(ordered):
        if i >= a.max_candidates:
            break
        got = execute(cand, a.url, a.rerank_url, cache)
        if a.gold_mode == "doc_f1":
            s = score_docs(got, row)
            if s <= base + a.min_gain:
                continue
        elif not covered(got, row, a.gold_mode, supporting):
            continue
        if first is None:
            first = {"repair_code": cand.render(tail, result_var), "repair_cost": list(cand.cost),
                     "candidates_tried": i + 1}
            if a.gold_mode == "doc_f1":
                first.update({"base_doc_f1": round(base, 4), "repair_doc_f1": round(s, 4)})
            if not a.full_scan:
                break        # cheapest-first, so the first hit is the cheapest repair
        all_covering.append({"code": cand.render(tail, result_var), "cost": list(cand.cost)})
    if first is None:
        return "no_repair", {"original_code": block.strip(), "original_cost": list(original.cost)}
    first.update({"original_code": block.strip(), "original_cost": list(original.cost)})
    if a.full_scan:
        first["all_covering"] = all_covering
    return "repaired", first


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dump", required=True, help="a TRAIN rollouts/<step>.jsonl")
    ap.add_argument("--url", default="http://127.0.0.1:8000/retrieve")
    ap.add_argument("--rerank_url", default="http://127.0.0.1:6980/rerank")
    ap.add_argument("--out", required=True)
    ap.add_argument("--gold_mode", choices=["answer_string", "supporting_passages", "doc_f1"],
                    default="answer_string",
                    help="doc_f1 (GlobalQA): score the retrieved set against `golden_doc_ids` and accept the "
                         "first candidate that improves on the original, rather than testing coverage")
    ap.add_argument("--min_gain", type=float, default=0.0,
                    help="doc_f1 mode: required improvement over the original before a repair is accepted")
    ap.add_argument("--ladder_lo", type=float, default=0.5,
                    help="relative ladder: lowest rung as a fraction of the emitted value. 0.5 clipped the "
                         "rerank width at its floor on 17.0%% of the first GlobalQA harvest's repairs")
    ap.add_argument("--ladder", choices=["global", "relative"], default="global",
                    help="relative centers the breadth/keep rungs on what the policy already wrote "
                         "(see local_ladder); use it where the useful edit is small, as on GlobalQA")
    ap.add_argument("--parquet", default="", help="required for supporting_passages")
    ap.add_argument("--limit", type=int, default=0,
                    help="cap the failed rollouts attempted, for a smoke run; 0 (default) attempts all, as every "
                         "reported round did")
    ap.add_argument("--max_candidates", type=int, default=300, help="grammar programs tried per row")
    ap.add_argument("--doc_budget", type=int, default=20)
    ap.add_argument("--measure_requirements", action="store_true",
                    help="score every row (not only failures) and record required vs emitted breadth/keep. "
                         "Answers whether a trained policy spends more where more is needed -- per-query "
                         "adaptivity measured on the policy, which offline coverage analysis cannot show")
    ap.add_argument("--workers", type=int, default=8,
                    help="rows processed concurrently. The work is blocking HTTP, so this is a near-linear "
                         "speedup, but the reranker is the binding resource -- starving it returns 0 docs "
                         "rather than erroring, which reads as a coverage failure")
    ap.add_argument("--repair_order", choices=["nearest", "cheapest", "breadth", "rerank"], default="nearest",
                    help="nearest = minimal edit from the policy's own program (default); breadth = "
                         "prefer candidates that change top_k, then minimal edit among those; rerank = "
                         "same but for adding/removing the rerank stage; cheapest = "
                         "absolute cost, which almost never selects fusion")
    ap.add_argument("--full_scan", action="store_true",
                    help="record every covering program, not just the cheapest; for analysis")
    ap.add_argument("--dedupe_questions", action=argparse.BooleanOptionalAction, default=True,
                    help="attempt each distinct question once. Coverage rates are per-question, so duplicate "
                         "rollouts of one question cost full retrieval and add no information")
    a = ap.parse_args()

    # Train-split guard: see the module docstring. A val dump would silently poison the SFT set.
    if "/val_rollouts/" in a.dump or "/rollouts/" not in a.dump:
        sys.exit(f"ABORT: --dump must be a TRAIN dump under rollouts/, got {a.dump}. Harvesting validation "
                 f"questions would invalidate every downstream comparison.")

    rows = [json.loads(line) for line in open(a.dump) if line.strip()]
    # Which rollout a repair came from, by position among the dump's non-empty lines: verify must splice the
    # repair into this exact trajectory, and a question has several rollouts (one per group sample and step).
    for i, r in enumerate(rows):
        r["_dump_line"] = i
    supporting = {}
    if a.gold_mode == "supporting_passages":
        if not a.parquet:
            sys.exit("--gold_mode=supporting_passages requires --parquet")
        supporting = load_supporting(a.parquet)
        rows = [r for r in rows if len(supporting.get(qkey(r.get("input")), [])) == 2]
        print(f"gold_mode=supporting_passages: {len(rows)} rows carry a 2-passage annotation")

    if a.measure_requirements:
        a.full_scan = True
        pool, why = rows, "all rows (requirements mode)"
    else:
        pool, why = [r for r in rows if r.get("score", 0.0) <= 0.5], "failed rollouts"
    usable = [r for r in pool if not has_var_query(r.get("output")) and last_turn(r.get("output"))]
    if a.dedupe_questions:
        # Each question appears once per rollout in the group, so a raw pool is ~3x duplicated.
        seen, deduped = set(), []
        for r in usable:
            q = qkey(r.get("input"))
            if q not in seen:
                seen.add(q)
                deduped.append(r)
        print(f"dedupe_questions: {len(usable)} replayable rows -> {len(deduped)} distinct questions")
        usable = deduped
    if a.limit:
        usable = usable[:a.limit]
    print(f"{len(rows)} rollouts; {len(pool)} {why}; {len(usable)} replayable and attempted")

    # Row-level concurrency. Every row is independent (`select_repair` builds its own cache) and the work
    # is pure blocking HTTP against the retriever and reranker, so threads pipeline it.
    counts: dict[str, int] = {}
    tried, saved_calls = [], []
    done = 0

    def _work(idx_row):
        idx, r = idx_row
        return idx, r, *select_repair(r, a, supporting)

    results: list = [None] * len(usable)
    with concurrent.futures.ThreadPoolExecutor(max_workers=a.workers) as pool:
        for idx, r, status, rec in pool.map(_work, enumerate(usable)):
            results[idx] = (r, status, rec)
            done += 1
            if done % 25 == 0:
                got = sum(1 for x in results if x and x[1] == "repaired")
                print(f"  {done}/{len(usable)} attempted, {got} repaired", flush=True)

    with open(a.out, "w") as fh:
        for r, status, rec in results:
            counts[status] = counts.get(status, 0) + 1
            if rec is not None:
                if a.measure_requirements and status == "repaired":
                    reqs = [min(int(x) for x in re.findall(r"_retrieval\([^)]*top_k=(\d+)\)", c["code"]) or [0])
                            for c in rec.get("all_covering", [])]
                    keeps = []
                    for c in rec.get("all_covering", []):
                        m2 = re.search(r"rerank\((?:[^()]|\([^()]*\))*top_k=(\d+)\)", c["code"])
                        if m2:
                            keeps.append(int(m2.group(1)))
                    orig = parse_program(last_turn(r.get("output")))
                    rec.update({"required_topk": min(reqs) if reqs else None,
                                "required_keep": min(keeps) if keeps else None,
                                "emitted_topk": max((c.top_k for c in orig.calls), default=None),
                                "emitted_keep": orig.rerank_keep,
                                "acc": r.get("score", 0.0), "n_covering": len(rec.get("all_covering", []))})
                    rec.pop("all_covering", None)      # per-row payload; the aggregates above are what we use
                rec.update({"status": status, "dump_line": r["_dump_line"], "uid": r.get("uid"),
                            "step": r.get("step"),
                            "question": qkey(r.get("input"))[:200], "gts": golds(r.get("gts")),
                            "gold_mode": a.gold_mode})
                fh.write(json.dumps(rec) + "\n")
            if status == "repaired":
                tried.append(rec["candidates_tried"])
                saved_calls.append(rec["original_cost"][0] - rec["repair_cost"][0])

    rep = counts.get("repaired", 0)
    # A retrieval-repair rate must exclude rows whose own program already covered: those failed to answer,
    # not to retrieve, and belong to the conversion branch instead.
    miss = rep + counts.get("no_repair", 0)
    # `why` names the pool actually used: requirements mode scores every row, so labelling the denominator
    # "failed rollouts" there would misread a whole-population rate as a failure rate.
    print(f"\nof {len(usable)} {why} attempted:")
    for k, n in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"  {k:<18}{n:>5}{n / len(usable):>8.1%}")
    print("  (all statuses are last-turn only: an earlier turn may have retrieved the gold)")
    print(f"\nretrieval misses (repaired + no_repair): {miss}")
    if miss:
        print(f"  repairable by the grammar: {rep}/{miss} = {rep / miss:.1%}")
    if tried:
        print(f"  candidates scanned before the cheapest hit: mean {sum(tried) / len(tried):.1f}, max {max(tried)}")
        cheaper = sum(1 for d in saved_calls if d > 0)
        print(f"  repairs using FEWER retrieval calls than the original: {cheaper}/{len(saved_calls)}")
    print(f"-> {a.out}")


if __name__ == "__main__":
    main()
