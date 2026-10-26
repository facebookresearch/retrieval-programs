# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for the prompt templates."""

import ast
import hashlib
import importlib.util
import os
import re
import unittest

import prompts
from prompts import create_prompt, create_rag_prompt

CODE = re.compile(r"<code>(.*?)</code>", re.DOTALL)

# Templates selected through `create_prompt`.
PROMPT_KEYS = [
    "rpl-openqa", "rpl-globalqa", "base", "autorefine", "direct", "cot", "ircot", "base-globalqa", "direct-globalqa",
    "cot-globalqa", "ircot-globalqa", "marag",
]
# Templates that need retrieved context and so go through `create_rag_prompt`.
RAG_KEYS = ["rag", "rag-globalqa"]
# Templates whose exemplars are Python.
CODE_KEYS = ["rpl-openqa", "rpl-globalqa", "marag"]

# Builtins the exemplars legitimately call.
_BUILTINS = {"print", "enumerate", "len", "sorted", "set", "list", "range", "min", "max", "sum", "str", "int"}


def _sdk_names():
    """Public callables the sandbox injects, read from the SDK itself rather than from a name list."""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "rpl", "search", "search_sdk.py")
    spec = importlib.util.spec_from_file_location("search_sdk", os.path.normpath(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return {n for n in dir(mod) if not n.startswith("_") and callable(getattr(mod, n))}


def _exemplars(text: str) -> list:
    """The Python exemplars in a prompt.

    `<code>(.*?)</code>` also matches the instruction sentence "write Python between <code> and </code>",
    whose captured body is the bare word "and". Drop that one span explicitly rather than skipping every
    block that fails to parse, so a genuinely malformed exemplar still raises.
    """
    return [b for b in CODE.findall(text) if b.strip() != "and"]


def _called_names(block: str) -> set:
    """Function names called in one exemplar, by parsing it rather than pattern-matching."""
    out = set()
    for node in ast.walk(ast.parse(block.strip())):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            out.add(node.func.id)
    return out


class RegistryTest(unittest.TestCase):
    def test_every_released_key_resolves(self):
        for key in PROMPT_KEYS:
            self.assertTrue(create_prompt("who wrote Beloved?", template_type=key).strip())

    def test_question_is_appended_exactly_once(self):
        for key in PROMPT_KEYS:
            out = create_prompt("who wrote Beloved?", template_type=key)
            self.assertEqual(out.count("Question: who wrote Beloved?"), 1, key)
            self.assertTrue(out.endswith("\n"), key)

    def test_unknown_key_raises(self):
        with self.assertRaises(NotImplementedError):
            create_prompt("q?", template_type="no-such-template")

    def test_rag_keys_refuse_create_prompt_and_work_through_create_rag_prompt(self):
        for key in RAG_KEYS:
            with self.assertRaises(ValueError):
                create_prompt("q?", template_type=key)
            out = create_rag_prompt("q?", "some passage", key)
            self.assertIn("Question: q?", out)
            self.assertIn("some passage", out)


class SdkConsistencyTest(unittest.TestCase):
    """Every primitive a prompt uses must exist, and every primitive it names must be demonstrated."""

    def setUp(self):
        self.sdk = _sdk_names()

    def test_exemplars_only_call_primitives_the_sdk_provides(self):
        for key in CODE_KEYS:
            text = create_prompt("q?", template_type=key)
            for block in _exemplars(text):
                for name in _called_names(block) - _BUILTINS:
                    self.assertIn(name, self.sdk,
                                  f"{key} exemplar calls {name}(), which search_sdk does not provide")


class GlobalQAPromptTest(unittest.TestCase):
    """`rpl-globalqa` is baked into the data of the reported GlobalQA runs, so its text must not drift."""

    def test_or_clause_exemplars_query_each_clause_separately(self):
        text = create_prompt("q", "rpl-globalqa")
        self.assertIn("retrieve each criterion with its own short query", text)
        self.assertIn('fuse(sparse_retrieval("Dreamweaver", top_k=5), sparse_retrieval("Revenue Cycle Advocate", top_k=10))', text)
        self.assertIn('fuse(sparse_retrieval("mainframe programmer", top_k=5), sparse_retrieval("COBOL developer", top_k=10))', text)
        self.assertEqual(len(_exemplars(text)), 4)

    def test_text_matches_the_trained_runs(self):
        """Pinned to the prompt the reported runs trained on. An intended edit means rebuilding the GlobalQA
        parquets and retraining; update the hash only then."""
        digest = hashlib.sha256(prompts._RPL_GLOBALQA_PROMPT.encode()).hexdigest()
        self.assertEqual(digest, "8fe67840756d4f39d36f267b24abfb6eca78c81b41f6ebd19e1c3cf3a26d0292")
