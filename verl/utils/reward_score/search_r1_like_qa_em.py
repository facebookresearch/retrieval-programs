# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 Search-R1 Contributors
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Adapted from https://github.com/PeterGriffinJin/Search-R1/blob/main/verl/utils/reward_score/qa_em.py

import random
import re
import string
from collections import Counter


def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def em_check(prediction, golden_answers):
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]
    normalized_prediction = normalize_answer(prediction)
    score = 0
    for golden_answer in golden_answers:
        golden_answer = normalize_answer(golden_answer)
        if golden_answer == normalized_prediction:
            score = 1
            break
    return score


def subem_check(prediction, golden_answers):
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]
    normalized_prediction = normalize_answer(prediction)
    score = 0
    for golden_answer in golden_answers:
        golden_answer = normalize_answer(golden_answer)
        if golden_answer in normalized_prediction:
            score = 1
            break
    return score


def cover_em_check(prediction, golden_answers):
    """1 if a normalized gold answer contains the prediction or the prediction contains a gold (either way)."""
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]
    norm_pred = normalize_answer(prediction)
    for golden_answer in golden_answers:
        norm_gold = normalize_answer(golden_answer)
        if norm_gold and (norm_gold in norm_pred or norm_pred in norm_gold):
            return 1
    return 0


def f1_check(prediction, golden_answers):
    """Max SQuAD-style token-level F1 between the normalized prediction and any golden answer."""
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]
    pred_tokens = normalize_answer(prediction).split()
    best = 0.0
    for golden_answer in golden_answers:
        gold_tokens = normalize_answer(golden_answer).split()
        if not pred_tokens or not gold_tokens:
            best = max(best, float(pred_tokens == gold_tokens))
            continue
        num_same = sum((Counter(pred_tokens) & Counter(gold_tokens)).values())
        if num_same == 0:
            continue
        precision, recall = num_same / len(pred_tokens), num_same / len(gold_tokens)
        best = max(best, 2 * precision * recall / (precision + recall))
    return best


def _extract_ids(text):
    """Parse the integer set from a GlobalQA answer string, e.g. '8' -> [8], '1183, 758, 719' -> [...]."""
    if text is None:
        return []
    return [int(x) for x in re.findall(r"-?\d+", str(text))]


def globalqa_em_check(prediction, golden_answers):
    """1.0 iff the prediction's integer set exactly equals a gold answer's (order-insensitive).

    GlobalQA answers are a count/id (Count, MinMax) or an id list (Sort, TopK) -- a deterministic
    aggregation of the gold doc set -- so exact correctness is set-equality on the parsed integers.
    """
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]
    pred = set(_extract_ids(prediction))
    for gold in golden_answers:
        g = set(_extract_ids(gold))
        if g and pred == g:
            return 1.0
    return 0.0


def globalqa_f1_check(prediction, golden_answers):
    """Max set-level F1 between the prediction's integer set and any gold answer's.

    The GlobalQA / MARAG-R1 headline answer metric: partial credit for partial coverage of the id set,
    degenerating to EM for the single-int Count/MinMax answers. Used as the (dense) train reward.
    """
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]
    pred = set(_extract_ids(prediction))
    best = 0.0
    for gold in golden_answers:
        g = set(_extract_ids(gold))
        if not pred or not g:
            continue
        tp = len(pred & g)
        if tp == 0:
            continue
        precision, recall = tp / len(pred), tp / len(g)
        best = max(best, 2 * precision * recall / (precision + recall))
    return best


def extract_solution(solution_str):
    """Extract the equation from the solution string."""
    # Remove everything before the first "Assistant:"
    # if "Assistant:" in solution_str:
    #     solution_str = solution_str.split("Assistant:", 1)[1]
    # elif "<|im_start|>assistant" in solution_str:
    #     solution_str = solution_str.split("<|im_start|>assistant", 1)[1]
    # else:
    #     return None
    # solution_str = solution_str.split('\n')[-1]

    answer_pattern = r"<answer>(.*?)</answer>"
    match = re.finditer(answer_pattern, solution_str, re.DOTALL)
    matches = list(match)

    # If there are 0  matches, return None
    if len(matches) < 1:
        return None

    # If there are 2 or more matches, return the last one
    return matches[-1].group(1).strip()


def count_answer_tags(text):
    opening_tags = text.count("<answer>")
    closing_tags = text.count("</answer>")

    return opening_tags, closing_tags


def compute_score(solution_str, ground_truth, method="strict", format_score=0.0, score=1.0):
    """The scoring function for exact match (EM).

    Args:
        solution_str: the solution text
        ground_truth: the ground truth
        method: the method to extract the solution, choices are 'strict' and 'flexible'
        format_score: the score for the format
        score: the score for the correct answer
    """
    answer = extract_solution(solution_str=solution_str)
    open_count, close_count = count_answer_tags(solution_str)
    do_print = random.randint(1, 64) == 1

    if do_print:
        print("--------------------------------")
        print(f"Golden answers: {ground_truth['target']}")
        if answer is not None:
            print(f"Extracted answer is not None: {answer}")
        else:
            print("Extracted answer: None!")
        print(f"Solution string: {solution_str}")

    if answer is None:
        return 0
    else:
        if em_check(answer, ground_truth["target"]):
            if open_count > 10 or close_count > 10:  # prevent output a lot of </answer>
                score = score / 4
                return score
            return score
        else:
            return format_score


def compute_score_subem(solution_str, ground_truth, method="strict", format_score=0.0, score=1.0):
    """The scoring function for substring exact match (EM).

    Args:
        solution_str: the solution text
        ground_truth: the ground truth
        method: the method to extract the solution, choices are 'strict' and 'flexible'
        format_score: the score for the format
        score: the score for the correct answer
    """
    answer = extract_solution(solution_str=solution_str)
    do_print = random.randint(1, 64) == 1

    if do_print:
        print("--------------------------------")
        print(f"Golden answers: {ground_truth['target']}")
        print(f"Extracted answer: {answer}")
        print(f"Solution string: {solution_str}")

    if answer is None:
        return 0
    else:
        if subem_check(answer, ground_truth["target"]):
            return score
        else:
            return format_score


# --- RPL/SaT train-only reward shaping ---
# SaT search observation: a NON-EMPTY <search> query immediately followed by <information>. Used only
# for the SaT fallback -- RPL gates on the sandbox's real search_calls. group 1 = query, group 2 = info.
_SAT_OBS = re.compile(r"<search>((?:(?!</search>).)*)</search>\s*<information>(.*?)</information>", re.DOTALL)
# A well-formed response has ALL text inside these blocks (nothing between tags) -- covers both the RPL
# (code/output) and SaT (search/information) action vocabularies.
_BLOCK = re.compile(
    r"<think>.*?</think>|<answer>.*?</answer>"
    r"|<code>.*?</code>|<output>.*?</output>|<search>.*?</search>|<information>.*?</information>",
    re.DOTALL,
)


def searched(response_str, search_calls=None):
    """True if the trajectory made >=1 real retrieval.

    RPL: the sandbox's `search_calls` count (ground truth). SaT (`search_calls` None -- no sandbox): a
    NON-EMPTY `<search>` query immediately followed by a non-empty `<information>` (the injected
    adjacency can't be faked in prose). Used by the conditional reward modes.
    """
    if search_calls is not None:
        return search_calls > 0
    return any(m.group(1).strip() and m.group(2).strip() for m in _SAT_OBS.finditer(response_str))


def well_formed(response_str):
    """True if the RESPONSE has a `<think>`, a real `<answer>` outside it, and ALL its text inside
    think/code/output/answer/search/information blocks (nothing between tags).

    Works for both RPL (`<code>`/`<output>`) and SaT (`<search>`/`<information>`). The reward manager
    pays its optional format bonus on this.
    """
    answer_outside_think = re.sub(r"<think>.*?</think>", "", response_str, flags=re.DOTALL)
    return (
        re.search(r"<think>.*?</think>", response_str, re.DOTALL) is not None
        and re.search(r"<answer>.*?</answer>", answer_outside_think, re.DOTALL) is not None
        and _BLOCK.sub("", response_str).strip() == ""
    )
