# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for the GlobalQA rows `globalqa_search.py` writes."""

import unittest

from globalqa_search import build_rows, infer_task_type


class BuildRowsTest(unittest.TestCase):
    def test_a_row_carries_the_graded_fields(self):
        (row,) = build_rows([{"question": "How many documents match X?", "answer": 3, "golden_doc_ids": ["5", 7]}],
                            "base-globalqa", "test")
        self.assertEqual(row["data_source"], "globalqa")
        self.assertEqual(row["reward_model"]["ground_truth"],
                         {"target": ["3"], "golden_doc_ids": [5, 7], "task_type": "count"})
        self.assertEqual(row["extra_info"], {"split": "test", "index": 0})
        self.assertTrue(row["prompt"][0]["content"].endswith(" Question: How many documents match X?\n"))

    def test_excluded_questions_are_dropped_without_renumbering_the_rest(self):
        items = [{"question": f"How many {x}?", "answer": 1, "golden_doc_ids": [1]} for x in "ABC"]
        rows = build_rows(items, "base-globalqa", "train", exclude=frozenset({"How many B?"}))
        self.assertEqual([r["extra_info"]["index"] for r in rows], [0, 2])

    def test_a_missing_answer_raises_instead_of_grading_against_nothing(self):
        with self.assertRaises(KeyError):
            build_rows([{"question": "How many?", "golden_doc_ids": []}], "base-globalqa", "test")


class InferTaskTypeTest(unittest.TestCase):
    def test_each_type_and_the_precedence_between_them(self):
        for question, want in (("How many resumes mention SQL?", "count"),
                               ("What are the top 2 smallest ids?", "topk"),
                               ("Sort the ids of matching documents.", "sort"),
                               ("What is the largest id?", "minmax"),
                               ("How many of the top 3 match?", "count"),
                               ("Which documents match?", "other")):
            self.assertEqual(infer_task_type(question), want, question)


if __name__ == "__main__":
    unittest.main()
