# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Build single-pass RAG parquets: retrieve once per question and freeze the passages into the prompt.

RAG is the one baseline whose prompt depends on data a template cannot supply, so the top-k passages are fetched
at prep time and baked into the parquet; the model never searches at inference. Rows are copied from an existing
data directory (e.g. one `qa_search.py` built) and only the prompt changes, so questions, answers and row order
match the other arms. Run it where the retriever is up and reachable (`scripts/serve_retriever.sh`).

Usage:
    python scripts/data_process/qa_rag.py --preset openqa --src data/nq_direct --out data/nq_rag
    python scripts/data_process/qa_rag.py --preset globalqa --src data/globalqa_base --out data/globalqa_rag
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
import requests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import prompts as P  # noqa: E402

PRESETS = {
    "openqa": {"template": "rag", "topk": 3},
    "globalqa": {"template": "rag-globalqa", "topk": 20},
}

# Byte-identical to `_passages2string`'s show_doc_id branch in rpl/llm_agent/rpl_agent_loop.py.
DOC_FMT_ID = "Doc {n}(id: {doc_id}, Title: {title}) {text}"


def question_of(content: str) -> str:
    """The question a baked prompt ends with. Raises rather than guessing -- a silently wrong split would
    retrieve for the wrong text and the arm would look like a retrieval failure."""
    _, sep, tail = content.rpartition(" Question: ")
    if not sep:
        raise ValueError(f"no ' Question: ' marker in prompt: {content[-120:]!r}")
    return tail[:-1] if tail.endswith("\n") else tail


def questions_of(df: pd.DataFrame) -> list[str]:
    """The retrieval queries: the `question` column where rows carry one (the FlashRAG sets do), else the question
    each baked prompt ends with (GlobalQA rows have no such column)."""
    if "question" in df.columns:
        return df["question"].tolist()
    return [question_of([dict(m) for m in list(p)][-1]["content"]) for p in df["prompt"]]


def _doc(item: dict) -> dict:
    return item["document"] if "document" in item else item


def render_openqa(result: list) -> str:
    """Format retriever rows as `Doc n(Title: <title line>) <passage>`, each ending in a newline.

    Upstream Search-R1's RAG format, with the corpus title line kept verbatim, quotes included. The
    paper's wiki-18 RAG baseline used exactly this format, so changing it changes that baseline.
    """
    parts = []
    for i, item in enumerate(result):
        title, _, text = _doc(item)["contents"].partition("\n")
        parts.append(f"Doc {i + 1}(Title: {title}) {text}\n")
    return "".join(parts)


def render_globalqa(result: list, doc_chars: int) -> str:
    """Format retriever rows with their corpus ids, each document cut to `doc_chars` (0 disables the cut).

    The cut is not optional. GlobalQA documents are full resumes, so 20 of them render to ~130k characters
    (~32k tokens) -- far past `data.max_prompt_length`, and `data.truncation=error` turns that into a hard failure
    on every row. The id and the head of each document are what the task needs (judge relevance, then report
    ids), and 600 characters x 20 documents is ~3k tokens.
    """
    lines = []
    for i, item in enumerate(result):
        doc = _doc(item)
        title, _, text = doc["contents"].partition("\n")
        title = title.strip('"')
        if doc_chars > 0:
            title, text = title[:doc_chars], text[:doc_chars]
        lines.append(DOC_FMT_ID.format(n=i + 1, doc_id=doc["id"], title=title, text=text))
    return "\n".join(lines)


def retrieve(url: str, queries: list[str], topk: int) -> list:
    r = requests.post(url, json={"queries": queries, "topk": topk, "return_scores": True}, timeout=600)
    r.raise_for_status()
    return r.json()["result"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preset", required=True, choices=sorted(PRESETS))
    ap.add_argument("--src", required=True, help="data directory whose rows are copied")
    ap.add_argument("--out", required=True)
    ap.add_argument("--splits", default="test", help="comma-separated <split>.parquet files to convert")
    ap.add_argument("--url", default="http://127.0.0.1:8000/retrieve")
    ap.add_argument("--topk", type=int, default=None, help="default: the preset's (openqa 3, globalqa 20)")
    ap.add_argument("--batch", type=int, default=32, help="questions per retriever request")
    ap.add_argument("--doc_chars", type=int, default=600,
                    help="globalqa only: per-document character cap, 0 disables")
    a = ap.parse_args()
    preset = PRESETS[a.preset]
    topk = preset["topk"] if a.topk is None else a.topk

    os.makedirs(a.out, exist_ok=True)
    for split in a.splits.split(","):
        df = pd.read_parquet(os.path.join(a.src, f"{split}.parquet"))
        questions = questions_of(df)
        contexts: list[str] = []
        for i in range(0, len(questions), a.batch):
            chunk = questions[i:i + a.batch]
            results = retrieve(a.url, chunk, topk)
            if len(results) != len(chunk):
                sys.exit(f"ABORT: the retriever answered {len(results)} of {len(chunk)} queries")
            contexts.extend(render_globalqa(r, a.doc_chars) if a.preset == "globalqa" else render_openqa(r)
                            for r in results)
            if (i // a.batch) % 20 == 0:
                print(f"  {split}: {i + len(chunk)}/{len(questions)} retrieved", flush=True)
        empty = sum(1 for c in contexts if not c.strip())
        if empty:
            sys.exit(f"ABORT: {empty}/{len(contexts)} contexts are empty -- the retriever returned nothing, "
                     f"which would score as 'RAG does not help' rather than as a broken prep.")
        if a.preset == "globalqa":
            noid = sum(1 for c in contexts if "(id: " not in c)
            if noid:
                sys.exit(f"ABORT: {noid}/{len(contexts)} contexts carry no document id; GlobalQA answers are ids.")
        new = []
        for prm, q, ctx in zip(df["prompt"], questions, contexts):
            msgs = [dict(m) for m in list(prm)]
            msgs[-1]["content"] = P.create_rag_prompt(q, ctx, preset["template"])
            new.append(np.array(msgs, dtype=object))
        df["prompt"] = new
        out = os.path.join(a.out, f"{split}.parquet")
        df.to_parquet(out, index=False)
        print(f"wrote {len(df)} rows (preset={a.preset}, topk={topk}) -> {out}")


if __name__ == "__main__":
    main()
