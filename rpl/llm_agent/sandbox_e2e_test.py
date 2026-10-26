# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import json
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer

from rpl.llm_agent.code_executor import CodeExecutor


class _Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        n = int(self.headers["Content-Length"])
        self.rfile.read(n)
        body = json.dumps(
            {"result": [[{"document": {"id": "1", "contents": '"Berlin"\nCapital.'}, "score": 0.9}]]}
        ).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):
        pass


class SandboxE2ETest(unittest.TestCase):
    def test_code_calls_search_through_sandbox(self):
        srv = HTTPServer(("127.0.0.1", 0), _Handler)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        try:
            url = f"http://127.0.0.1:{srv.server_address[1]}/retrieve"
            ex = CodeExecutor(retriever_url=url, timeout_sec=10)
            code = 'for d in search("capital of Germany", top_k=1):\n    print(d["title"], d["text"])'
            results = ex.run_batch_detailed([code])
        finally:
            srv.shutdown()
            srv.server_close()
        self.assertIn("Berlin", results[0].observation())
        self.assertEqual(results[0].search_calls, 1)

    def test_code_uses_fuse_and_dedup_in_sandbox(self):
        srv = HTTPServer(("127.0.0.1", 0), _Handler)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        try:
            url = f"http://127.0.0.1:{srv.server_address[1]}/retrieve"
            ex = CodeExecutor(retriever_url=url, timeout_sec=10)
            # dense + sparse both return doc id "1" -> fuse dedups to a single doc. Confirms fuse/dedup are
            # exposed in the sandbox namespace and do NOT charge the search-call budget (only the 2 retrievals do).
            code = (
                'docs = fuse(dense_retrieval("q", top_k=1), sparse_retrieval("q", top_k=1))\n'
                'print(len(dedup(docs)), docs[0]["title"])'
            )
            results = ex.run_batch_detailed([code])
        finally:
            srv.shutdown()
            srv.server_close()
        self.assertIn("1 Berlin", results[0].observation())
        self.assertEqual(results[0].search_calls, 2)


class SandboxStateTest(unittest.TestCase):
    """Cross-block variable reuse. Without the persisted state channel every such reference is a NameError,
    because each <code> block executes in a fresh namespace."""

    def _executor(self, srv):
        return CodeExecutor(retriever_url=f"http://127.0.0.1:{srv.server_address[1]}/retrieve", timeout_sec=10)

    def _serve(self):
        srv = HTTPServer(("127.0.0.1", 0), _Handler)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        return srv

    def test_variable_does_not_carry_over_without_state(self):
        """Default (state not threaded) keeps the fresh-namespace behavior."""
        srv = self._serve()
        try:
            ex = self._executor(srv)
            ex.run_batch_detailed(['docs = search("capital of Germany", top_k=1)'])
            second = ex.run_batch_detailed(['print(len(docs))'])[0]
        finally:
            srv.shutdown(); srv.server_close()
        self.assertIn("NameError", second.observation())

    def test_variable_carries_over_when_state_is_threaded(self):
        srv = self._serve()
        try:
            ex = self._executor(srv)
            first = ex.run_batch_detailed(['docs = search("capital of Germany", top_k=1)'])[0]
            self.assertIn("docs", first.state)
            second = ex.run_batch_detailed(['print(docs[0]["title"])'], [first.state])[0]
        finally:
            srv.shutdown(); srv.server_close()
        self.assertIsNone(second.error)
        self.assertIn("Berlin", second.observation())

    def test_primitives_win_over_a_shadowing_variable(self):
        """A model that assigns to `rerank` must not break rerank in the NEXT block."""
        srv = self._serve()
        try:
            ex = self._executor(srv)
            first = ex.run_batch_detailed(['rerank = [1, 2, 3]'])[0]
            second = ex.run_batch_detailed(['print(callable(rerank))'], [first.state])[0]
        finally:
            srv.shutdown(); srv.server_close()
        self.assertEqual(second.observation().strip(), "True")

    def test_unserialisable_values_are_dropped_not_fatal(self):
        srv = self._serve()
        try:
            ex = self._executor(srv)
            r = ex.run_batch_detailed(['keep = [1, 2]\ngen = (i for i in range(3))'])[0]
        finally:
            srv.shutdown(); srv.server_close()
        self.assertIsNone(r.error)
        self.assertIn("keep", r.state)
        self.assertNotIn("gen", r.state)

    def test_oversized_state_is_dropped_loudly(self):
        srv = self._serve()
        try:
            ex = self._executor(srv)
            ex.max_state_bytes = 64 * 1024  # small, so the block stays cheap
            r = ex.run_batch_detailed([f'big = "x" * {ex.max_state_bytes * 2}'])[0]
        finally:
            srv.shutdown(); srv.server_close()
        self.assertEqual(r.state, {})
        self.assertIn("exceeded", r.observation())

    def test_concurrent_blocks_do_not_share_state(self):
        """Worker reuse must not leak one rollout's namespace into another's.

        This is the failure mode that every other test here would still pass through: `run_batch_detailed`
        runs blocks on a thread pool over a shared worker pool, so a namespace held in the worker (rather
        than threaded by the caller) would cross-contaminate concurrent rollouts.
        """
        srv = self._serve()
        try:
            ex = self._executor(srv)
            n = 12
            codes = ['print(tag)'] * n
            states = [{"tag": f"rollout-{i}"} for i in range(n)]
            results = ex.run_batch_detailed(codes, states)
        finally:
            srv.shutdown(); srv.server_close()
        for i, r in enumerate(results):
            self.assertIsNone(r.error, f"block {i}: {r.error}")
            self.assertEqual(r.observation().strip(), f"rollout-{i}")

    def test_state_does_not_persist_in_the_worker_between_calls(self):
        """A variable set in one block must be gone in the next unless the caller threads it back."""
        srv = self._serve()
        try:
            ex = self._executor(srv)
            ex.run_batch_detailed(['leaked = "should not survive"'])
            after = ex.run_batch_detailed(['print(leaked)'])[0]
        finally:
            srv.shutdown(); srv.server_close()
        self.assertIn("NameError", after.observation())
