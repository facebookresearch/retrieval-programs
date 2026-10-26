# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for `qa_rag`'s context formats, which decide the RAG baselines' numbers."""

import unittest

import pandas as pd

import prompts as P
from qa_rag import questions_of, render_globalqa, render_openqa


class RenderOpenqaTest(unittest.TestCase):
    def test_the_title_line_is_kept_verbatim_and_each_document_ends_its_line(self):
        docs = [{"document": {"id": "1", "contents": '"Nobel Prize in Physics"\nThe Nobel Prize in Physics () is'},
                 "score": 0.9},
                {"id": "2", "contents": "Onikwu\nThe dry season"}]   # a bare row, as with return_scores=False
        self.assertEqual(render_openqa(docs),
                         'Doc 1(Title: "Nobel Prize in Physics") The Nobel Prize in Physics () is\n'
                         'Doc 2(Title: Onikwu) The dry season\n')

    def test_the_question_precedes_the_context(self):
        prompt = P.create_rag_prompt("who won?", render_openqa([{"contents": "T\nx"}]), "rag")
        self.assertTrue(prompt.endswith(" Question: who won? Context: Doc 1(Title: T) x\n \n"))


class RenderGlobalqaTest(unittest.TestCase):
    def test_ids_are_shown_and_each_part_is_cut(self):
        docs = [{"document": {"id": 7, "contents": '"Analyst"\n' + "a" * 900}},
                {"document": {"id": 8, "contents": "b" * 700}}]   # no newline: the whole head is the title
        self.assertEqual(render_globalqa(docs, 600),
                         "Doc 1(id: 7, Title: Analyst) " + "a" * 600 + "\nDoc 2(id: 8, Title: " + "b" * 600 + ") ")

    def test_a_zero_cap_disables_the_cut(self):
        self.assertEqual(render_globalqa([{"id": 1, "contents": "t\n" + "c" * 900}], 0),
                         "Doc 1(id: 1, Title: t) " + "c" * 900)


class QuestionsOfTest(unittest.TestCase):
    def test_the_question_column_wins_and_the_prompt_is_the_fallback(self):
        prompt = [{"role": "user", "content": "Answer it. Question: how many?\n"}]
        self.assertEqual(questions_of(pd.DataFrame({"question": ["q?"], "prompt": [prompt]})), ["q?"])
        self.assertEqual(questions_of(pd.DataFrame({"prompt": [prompt]})), ["how many?"])


if __name__ == "__main__":
    unittest.main()
