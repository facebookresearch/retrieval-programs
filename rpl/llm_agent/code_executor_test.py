# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import importlib.util
import unittest

from rpl.llm_agent.code_executor import CodeExecutor


class CodeExecutorTest(unittest.TestCase):
    def setUp(self):
        self.ex = CodeExecutor(retriever_url="http://127.0.0.1:9/none", timeout_sec=5, max_workers=4)

    def test_prints_are_captured(self):
        r = self.ex.run_one("print('hello'); print(1 + 1)")
        self.assertIsNone(r.error)
        self.assertFalse(r.timed_out)
        self.assertIn("hello", r.stdout)
        self.assertIn("2", r.stdout)

    def test_exception_returns_traceback(self):
        r = self.ex.run_one("raise ValueError('boom')")
        self.assertIsNotNone(r.error)
        self.assertIn("ValueError: boom", r.error)

    def test_timeout_is_killed(self):
        r = CodeExecutor(retriever_url="x", timeout_sec=1).run_one("while True: pass")
        self.assertTrue(r.timed_out)

    def test_empty_block(self):
        self.assertEqual(self.ex.run_one("   ").error, "empty code block")

    @unittest.skipUnless(importlib.util.find_spec("pandas"), "needs numpy and pandas")
    def test_numpy_and_pandas_import_under_the_memory_cap(self):
        # Without pinned BLAS threads these imports hung until the wall clock (600 s in training), so every
        # numpy/pandas block came back as "execution timed out" and stalled its whole step.
        code = "import numpy as np; import pandas as pd; print(np.arange(4).sum(), len(pd.DataFrame({'a': [1, 2]})))"
        r = self.ex.run_one(code)
        self.assertFalse(r.timed_out)
        self.assertIsNone(r.error)
        self.assertEqual(r.stdout.strip(), "6 2")

    def test_batch_order_preserved(self):
        obs = self.ex.run_batch(["print('a')", "print('b')", "print('c')"])
        self.assertEqual([o.strip() for o in obs], ["a", "b", "c"])

    def test_external_ip_blocked(self):
        # Model code connecting to an external IP is refused by the sandbox network guard.
        r = self.ex.run_one("import socket; socket.create_connection(('8.8.8.8', 53), timeout=3)")
        self.assertIsNotNone(r.error)
        self.assertIn("external network blocked", r.error)

    def test_external_host_blocked(self):
        # External DNS lookups are refused before they can hit the network (no slow timeout).
        r = self.ex.run_one("import socket; socket.getaddrinfo('example.com', 80)")
        self.assertIsNotNone(r.error)
        self.assertIn("external network blocked", r.error)

    def test_loopback_allowed(self):
        # Loopback (the retriever endpoint) must reach the OS: nothing listens on :9 so it is refused,
        # but NOT by the guard -- so the retriever at 127.0.0.1 keeps working.
        r = self.ex.run_one("import socket; socket.create_connection(('127.0.0.1', 9), timeout=2)")
        self.assertIsNotNone(r.error)
        self.assertNotIn("external network blocked", r.error)


class SandboxEnvExportTest(unittest.TestCase):
    def test_rerank_and_marag_knobs_reach_the_sandbox(self):
        ex = CodeExecutor(
            retriever_url="http://127.0.0.1:9/none",
            rerank_relevance_threshold=0.42,
            marag_filter_frac=0.0,
            marag_topk=17,
        )
        env = ex._env()
        self.assertEqual(env["RERANK_RELEVANCE_THRESHOLD"], "0.42")
        self.assertEqual(env["MARAG_FILTER_FRAC"], "0.0", "predicate mode (frac=0) must survive the hop")
        self.assertEqual(env["MARAG_TOPK"], "17")

    def test_the_sandbox_child_actually_reads_them(self):
        """End-to-end through the fork, not just the dict: the child must see the exported values."""
        ex = CodeExecutor(retriever_url="http://127.0.0.1:9/none", timeout_sec=30, marag_topk=17)
        r = ex.run_one("import os; print(os.environ.get('MARAG_TOPK'), os.environ.get('MARAG_FILTER_FRAC'))")
        self.assertIsNone(r.error)
        self.assertIn("17", r.stdout)
