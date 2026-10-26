# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Prompt templates.

Two task settings, each with an RPL prompt and its baselines:
  * open-domain QA over Wikipedia -- `rpl-openqa`; baselines `direct`, `cot`, `rag`, `ircot`, `base` (SaT),
    `autorefine`.
  * GlobalQA corpus aggregation -- `rpl-globalqa`; baselines `direct-globalqa`, `cot-globalqa`,
    `rag-globalqa`, `ircot-globalqa`, `base-globalqa` (SaT), `marag`.
"""


# base: the Search-R1 prompt. The policy emits a query between <search> tags and the retriever returns
# passages between <information> tags.
_BASE_PROMPT: str = """\
Answer the given question. You must conduct reasoning inside <think> and </think> first every time you get new information. \
After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query </search> and it will return the top searched results between <information> and </information>. \
You can search as many times as your want. If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations. For example, <answer> Seoul </answer>.
"""

# autorefine: AutoRefine (arXiv 2505.11277), which adds a <refine> step that distills retrieved passages
# before answering. Reimplemented here as prompt + reward only, so it shares our retriever and backbone.
_AUTOREFINE_PROMPT: str = """\
Answer the given question. You must conduct reasoning inside <think> and </think> first every time you get new information. \
After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query </search> and it will return the top searched results between <information> and </information>. \
Every time you receive search results, you must distill the parts that help answer the question inside <refine> and </refine>, keeping the relevant facts verbatim and discarding the rest. \
You can search as many times as your want. If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, based on your refined knowledge and without detailed illustrations. For example, <answer> Seoul </answer>.
"""

# direct: closed-book, answer immediately. No retrieval.
_DIRECT_PROMPT: str = """\
Answer the given question. Provide the answer inside <answer> and </answer>, without detailed illustrations. For example, <answer> Seoul </answer>."""

# cot: closed-book with chain-of-thought reasoning before the answer. No retrieval.
_COT_PROMPT: str = """\
Answer the given question. \
You should first have a reasoning process in mind and then provide the answer. \
Show your reasoning in <think> </think> tags and return the final answer in <answer> </answer> tags, for example <answer> Seoul </answer>."""

# rag: single-pass retrieval -- the top-k passages are fetched once at data-prep time and frozen into the
# parquet, so the model never searches at inference. Built through `create_rag_prompt`.
_RAG_PROMPT: str = """\
Answer the given question with some potentially useful context. \
You should analyze the question carefully, evaluate the given context (which may or may not be useful), and then generate an accurate and well-reasoned response. \
You should first have a reasoning process in mind and then provide the answer. \
Show your reasoning in <think> </think> tags and return the final answer in <answer> </answer> tags, for example <answer> Seoul </answer>."""

# ircot: IRCoT -- interleave one reasoning step with one retrieval step.
_IRCOT_PROMPT: str = """\
Answer the given question by interleaving reasoning with retrieval. \
Work one step at a time: think about what single fact you still need, then look it up before continuing. \
Put each reasoning step between <think> and </think>. \
To look something up, put a query between <search> and </search>, and the top documents will be returned between <information> and </information>. \
Base your next reasoning step on what those documents say, then search again for the next missing fact. \
Begin by searching for the question itself. \
When the reasoning is sufficient to answer, return the final answer in <answer> </answer> tags, for example <answer> Seoul </answer>."""

# rpl-openqa: the RPL prompt for open-domain QA. The policy writes Python that composes the retrieval SDK
# (dense_retrieval / sparse_retrieval / rerank / fuse / dedup) instead of emitting a fixed search query.
_RPL_OPENQA_PROMPT: str = """\
Answer the given question. You must conduct reasoning inside <think> and </think> first every time you get new information. After reasoning, if you find you lack some knowledge, you can write Python between <code> and </code> to search an external knowledge corpus. \
The following functions are available (do not import any modules or define or call any other function):
- dense_retrieval(query: str, top_k: int): semantic search; best for paraphrased or conceptual questions (for example "what is", "who is", descriptions).
- sparse_retrieval(query: str, top_k: int): keyword search; best for exact names, numbers, dates, rare terms, or quoted strings.
- fuse(*doc_lists): merge several retrieved lists into one deduplicated, re-ranked list (reciprocal rank fusion).
- dedup(docs): remove duplicate documents from a single list.
- rerank(query: str, docs, top_k: int): reorder docs by relevance to the query with a cross-encoder and keep the top_k. Use it to turn a large, noisy candidate set into a short, focused one before reading.
Each retrieval returns a list of documents, each a dict with "title" and "text". Use print(...) to display what you want to read, and the printed output will be returned between <output> and </output>. A strong strategy is to retrieve BROADLY for recall (a large top_k), then rerank down to the few most relevant documents so your context stays focused and the answer is easy to find. \
For a conceptual question, a small dense retrieval may be enough:
<code>
for i, doc in enumerate(dense_retrieval("what causes the aurora borealis", top_k=3)):
    print(f"Doc {i+1}(Title: {doc['title']}) {doc['text']}")
</code>
For an exact name, number, or rare term, use sparse_retrieval:
<code>
for i, doc in enumerate(sparse_retrieval("Treaty of Westphalia 1648 signatories", top_k=5)):
    print(f"Doc {i+1}(Title: {doc['title']}) {doc['text']}")
</code>
When a single retriever should surface the answer but its top hits are noisy, retrieve broadly from it alone and rerank down:
<code>
docs = dense_retrieval("what is the main cause of coral bleaching", top_k=20)
for i, doc in enumerate(rerank("what is the main cause of coral bleaching", docs, top_k=3)):
    print(f"Doc {i+1}(Title: {doc['title']}) {doc['text']}")
</code>
When the answer is hard to find, retrieve a LARGE candidate set from both retrievers, then rerank to keep only the most relevant few:
<code>
candidates = fuse(dense_retrieval("who wrote the novel Beloved", top_k=20), sparse_retrieval("Beloved novel author", top_k=20))
for i, doc in enumerate(rerank("who wrote the novel Beloved", candidates, top_k=3)):
    print(f"Doc {i+1}(Title: {doc['title']}) {doc['text']}")
</code>
For a multi-part or multi-hop question, decompose it into sub-queries, retrieve each, fuse, then rerank against the full question to surface the documents that answer it:
<code>
results = []
for q in ["director of the film Inception", "birthplace of Christopher Nolan"]:
    results.append(dense_retrieval(q, top_k=20))
    results.append(sparse_retrieval(q, top_k=20))
for i, doc in enumerate(rerank("where was the director of the film Inception born", fuse(*results), top_k=5)):
    print(f"Doc {i+1}(Title: {doc['title']}) {doc['text']}")
</code>
You can write code as many times as you want. If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations. For example, <answer> Seoul </answer>."""

# base-globalqa: SaT baseline for the GlobalQA corpus-aggregation task (Count / MinMax / Sort / TopK over
# the documents matching a criterion). The answer is a number or a comma-separated list of document ids.
_BASE_GLOBALQA_PROMPT: str = """\
Answer the given question about a document corpus. Some questions ask you to COUNT the documents matching a criterion, or to find the SMALLEST/LARGEST document id, SORT the ids, or list the TOP-K ids among the matching documents. \
You must conduct reasoning inside <think> and </think> first every time you get new information. \
After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query </search> and it will return the top searched results between <information> and </information>. \
You can search as many times as you want. When you know the answer, provide it inside <answer> and </answer> as the number or the comma-separated document ids, without detailed illustrations. For example, <answer> 8 </answer> or <answer> 330, 493 </answer>.
"""

# direct-globalqa: closed-book GlobalQA. Same task framing and answer format as `base-globalqa`, with the
# retrieval protocol removed.
_DIRECT_GLOBALQA_PROMPT: str = """\
Answer the given question about a document corpus. Some questions ask you to COUNT the documents matching a criterion, or to find the SMALLEST/LARGEST document id, SORT the ids, or list the TOP-K ids among the matching documents. When you know the answer, provide it inside <answer> and </answer> as the number or the comma-separated document ids, without detailed illustrations. For example, <answer> 8 </answer> or <answer> 330, 493 </answer>.
"""

# cot-globalqa: closed-book GlobalQA with chain-of-thought reasoning.
_COT_GLOBALQA_PROMPT: str = """\
Answer the given question about a document corpus. Some questions ask you to COUNT the documents matching a criterion, or to find the SMALLEST/LARGEST document id, SORT the ids, or list the TOP-K ids among the matching documents. You should first have a reasoning process in mind and then provide the answer, showing your reasoning inside <think> and </think>. When you know the answer, provide it inside <answer> and </answer> as the number or the comma-separated document ids, without detailed illustrations. For example, <answer> 8 </answer> or <answer> 330, 493 </answer>.
"""

# ircot-globalqa: IRCoT on GlobalQA. Shares the task framing and answer format with `base-globalqa` and
# adds only the interleaving protocol, so the comparison isolates the protocol.
_IRCOT_GLOBALQA_PROMPT: str = """\
Answer the given question about a document corpus. Some questions ask you to COUNT the documents matching a criterion, or to find the SMALLEST/LARGEST document id, SORT the ids, or list the TOP-K ids among the matching documents. Answer by interleaving reasoning with retrieval. Work one step at a time: think about what single fact you still need, then look it up before continuing. Put each reasoning step between <think> and </think>. To look something up, put a query between <search> and </search>, and the top documents will be returned between <information> and </information>. Base your next reasoning step on what those documents say, then search again for the next missing fact. Begin by searching for the question itself. When you know the answer, provide it inside <answer> and </answer> as the number or the comma-separated document ids, without detailed illustrations. For example, <answer> 8 </answer> or <answer> 330, 493 </answer>.
"""

# rag-globalqa: single-pass retrieval on GlobalQA. Built through `create_rag_prompt`.
_RAG_GLOBALQA_PROMPT: str = """\
Answer the given question about a document corpus. Some questions ask you to COUNT the documents matching a criterion, or to find the SMALLEST/LARGEST document id, SORT the ids, or list the TOP-K ids among the matching documents. You are given the documents retrieved for this question below; each is shown with its corpus id. Evaluate them (they may or may not be useful) and reason inside <think> and </think>. When you know the answer, provide it inside <answer> and </answer> as the number or the comma-separated document ids, without detailed illustrations. For example, <answer> 8 </answer> or <answer> 330, 493 </answer>.
"""

# rpl-globalqa: the RPL prompt for GlobalQA. `rerank(query, docs, top_k)` keeps a fixed number of documents
# chosen per question. The TOP-K and MIN/MAX exemplars query each OR-clause separately at a depth sized to it,
# because the policy copies exemplar literals: with one query per question at top_k=20 it kept that form.
_RPL_GLOBALQA_PROMPT: str = """\
Answer the given question about a document corpus. Each document has an integer "id". Many questions ask you to COUNT the documents matching a criterion, or to find the SMALLEST/LARGEST id, SORT the ids, or list the TOP-K ids among the matching documents -- so the answer is computed from the set of matching document ids. You must conduct reasoning inside <think> and </think> first, then write Python between <code> and </code> to search the corpus. The following functions are available (do not import any modules or define or call any other function):
- dense_retrieval(query: str, top_k: int = 3): semantic search; best for paraphrased or conceptual criteria.
- sparse_retrieval(query: str, top_k: int = 3): keyword search; best for exact skills, titles, or rare terms.
- fuse(*doc_lists): merge several retrieved lists into one deduplicated, re-ranked list (reciprocal rank fusion).
- dedup(docs): remove duplicate documents from a single list.
- rerank(query: str, docs: list, top_k: int): re-score documents against the query with a cross-encoder and keep only the top_k most relevant; use it to drop non-matching documents from a broad retrieval. The documents it keeps are the ones your answer is computed from, so choosing top_k too small discards matches and makes a COUNT too low; 10, 20 or 50 are reasonable starting points.
Each retrieval returns a list of documents, best match first, each a dict with "id", "title", "text", and "score". Scores are only comparable WITHIN one list -- fuse() re-scores by rank, so its numbers are much smaller than a single retriever's. Use print(...) to display what you want to read; the printed output is returned between <output> and </output>. The answer depends on exactly the set of documents that match the criterion, so the top_k you choose IS the answer set: too small and you miss matches, too large and unrelated documents inflate a COUNT and corrupt a MinMax or SORT. There is no single right value -- pick one that fits how many documents the question is likely to match, for example 10, 20, 50, or 100. When you are unsure, retrieve a moderate top_k first, print the titles to see where the results stop being relevant, then retrieve again with an adjusted top_k before answering. When the question joins several criteria with OR, retrieve each criterion with its own short query and a depth that fits how many documents THAT criterion is likely to match -- a specific skill or title often matches only a handful, so 5 or 10 -- and fuse them; a larger depth mostly adds documents that match nothing. For a COUNT question, use len:
<code>
candidates = fuse(dense_retrieval("aviation QC inspector with in-process inspection expertise", top_k=20), sparse_retrieval("aviation quality control inspector in-process inspection", top_k=20))
ids = {d["id"] for d in rerank("aviation quality control inspector with in-process inspection expertise", candidates, top_k=20)}
print("count:", len(ids))
</code>
For a SMALLEST/LARGEST question, use min or max over the ids:
<code>
docs = fuse(sparse_retrieval("Dreamweaver", top_k=5), sparse_retrieval("Revenue Cycle Advocate", top_k=10))
ids = sorted({d["id"] for d in docs})
print("smallest id:", ids[0], "largest id:", ids[-1])
</code>
For a SORT question, sort the matching ids:
<code>
candidates = fuse(dense_retrieval("healthcare recruiters with full-cycle recruiting", top_k=20), sparse_retrieval("healthcare recruiter full cycle recruiting", top_k=20))
print(sorted({d["id"] for d in rerank("healthcare recruiters with full-cycle recruiting", candidates, top_k=20)}))
</code>
For a TOP-K question (for example the 2 smallest ids), sort then slice:
<code>
docs = fuse(sparse_retrieval("mainframe programmer", top_k=5), sparse_retrieval("COBOL developer", top_k=10))
ids = sorted({d["id"] for d in docs})
print("top 2 smallest:", ids[:2])
</code>
You can write code as many times as you want. When you know the answer, give it inside <answer> and </answer> as the number or the comma-separated ids, without detailed illustrations. For example, <answer> 8 </answer> or <answer> 330, 493 </answer>."""

# marag: MARAG-R1 (arXiv 2510.27569) reimplemented as a baseline, teacher-free.
_MARAG_PROMPT: str = """\
Answer the given question using a corpus of documents. You must conduct reasoning inside <think> and </think> first, \
then call EXACTLY ONE tool by writing Python between <code> and </code>. \
The following tools are available (do not import any modules or define or call any other function):
- DenseSearch(query: str): semantic search; best for paraphrased or conceptual criteria.
- KeywordSearch(query: str): keyword search; best for exact names, rare terms, or quoted strings.
- DocumentFilter(query: str, docs: list): keep only the documents relevant to the query, dropping the rest.
- Python expressions over the documents you have gathered, for counting, sorting, set union and intersection.
Each search returns a list of documents, each a dict with "id", "title" and "text". The number of documents \
returned is fixed and is not yours to choose. Variables you assign persist across turns, so gather evidence \
one tool call at a time and combine it later. Use print(...) to display what you want to read; the printed \
output is returned between <output> and </output>. Call one tool per turn, for example:
<code>
a = DenseSearch("director of the film Inception")
print(a)
</code>
Then, in a later turn, a different tool:
<code>
b = KeywordSearch("Inception 2010 Christopher Nolan")
print(b)
</code>
Then narrow what you gathered:
<code>
kept = DocumentFilter("where was the director of Inception born", a + b)
print(kept)
</code>
Then combine or aggregate with Python:
<code>
ids = sorted({d["id"] for d in kept})
print("count:", len(ids), "ids:", ids)
</code>
You can take as many turns as you want. When you know the answer, give it inside <answer> and </answer>, \
without detailed illustrations. For example, <answer> Seoul </answer>."""


def create_prompt(question: str, template_type: str) -> str:
    """Return the full user prompt for `question` under `template_type`.

    Raises `NotImplementedError` for an unrecognized `template_type`, and `ValueError` for the
    single-pass retrieval templates, which need context and so must go through `create_rag_prompt`.
    """
    match template_type:
        case "rpl-openqa": prompt = _RPL_OPENQA_PROMPT
        case "rpl-globalqa": prompt = _RPL_GLOBALQA_PROMPT
        case "base": prompt = _BASE_PROMPT
        case "autorefine": prompt = _AUTOREFINE_PROMPT
        case "direct": prompt = _DIRECT_PROMPT
        case "cot": prompt = _COT_PROMPT
        case "ircot": prompt = _IRCOT_PROMPT
        case "base-globalqa": prompt = _BASE_GLOBALQA_PROMPT
        case "direct-globalqa": prompt = _DIRECT_GLOBALQA_PROMPT
        case "cot-globalqa": prompt = _COT_GLOBALQA_PROMPT
        case "ircot-globalqa": prompt = _IRCOT_GLOBALQA_PROMPT
        case "marag": prompt = _MARAG_PROMPT
        case "rag" | "rag-globalqa":
            raise ValueError(f"{template_type!r} needs retrieved context -- call `create_rag_prompt` instead")
        case _:
            raise NotImplementedError(f"Unrecognized template type: {template_type}")
    # rstrip so templates ending in "." or a trailing newline join uniformly as "... Question: <q>".
    return f"{prompt.rstrip()} Question: {question}\n"


def create_rag_prompt(question: str, context: str, template: str = "rag") -> str:
    """Return the single-pass retrieval prompt for `question` with `context` already retrieved.

    Separate from `create_prompt` because RAG is the one baseline whose prompt depends on data the template
    alone cannot supply: the top-k passages are fetched once at data-prep time and frozen into the parquet,
    so the model never searches at inference.
    """
    base = {"rag": _RAG_PROMPT, "rag-globalqa": _RAG_GLOBALQA_PROMPT}[template]
    return f"{base.rstrip()} Question: {question} Context: {context} \n"
