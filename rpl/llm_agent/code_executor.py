# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Subprocess sandbox for executing model-generated retrieval program blocks.

`CodeExecutor.run_batch(codes)` runs each code string in a Python subprocess with OS resource limits +
a wall-clock timeout, in parallel via a thread pool, and returns one observation string per input. The
child reaches the retriever only through the search SDK primitives.

Two backends (selected by SANDBOX_POOL):
  - one-shot (default): fork+exec a fresh subprocess per code block. Simple and robust (~50-100ms/block).
  - pool (SANDBOX_POOL=1): keep warm workers alive and stream code over length-framed stdin/stdout,
    eliminating the per-block fork+exec+import. A hung block is killed by the parent timeout (that worker
    is dropped, not reused); a dead worker is replaced on the next request.
"""
from __future__ import annotations

import json
import os
import queue
import signal
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

_RUNNER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sandbox_runner.py")


@dataclass
class ExecResult:
    stdout: str
    error: str | None
    timed_out: bool
    search_calls: int = 0
    dense_calls: int = 0       # per-method retrieval counts (complementary-search reward + routing analytics)
    sparse_calls: int = 0
    search_time: float = 0.0   # retriever HTTP time (s) for this block (latency profiling)
    run_time: float = 0.0      # parent-observed wall time for this block (s)
    rerank_calls: int = 0      # rerank() calls
    call_docs: list = field(default_factory=list)         # per retrieval call: [doc_contents, ...] (recall/precision)
    rerank_call_docs: list = field(default_factory=list)  # per rerank() call: [doc_contents, ...] (rerank precision)
    call_doc_ids: list = field(default_factory=list)      # per retrieval call: [corpus_id, ...] (GlobalQA doc F1)
    # JSON-serializable user variables surviving this block, to seed the NEXT block of the same rollout.
    # The CALLER owns and threads it, so there is no worker affinity or eviction to manage here.
    state: dict = field(default_factory=dict)

    def observation(self) -> str:
        if self.timed_out:
            return "execution timed out"
        if self.error:
            return self.error
        return self.stdout


@dataclass
class CodeExecutor:
    retriever_url: str
    # Cross-encoder rerank endpoint for the SDK `rerank()` primitive.
    rerank_url: str = "http://127.0.0.1:6980/rerank"
    # Optional endpoint JSON published by a standalone reranker server.
    # When set, the concrete rerank URL is re-resolved from it with a short TTL, so a restarted reranker (new
    # task IP) is picked up by a running training job without a relaunch. Empty -> the static `rerank_url`.
    rerank_endpoint_file: str = ""
    # Same idea for the retriever: re-resolve so a restarted retriever (new task IP) is picked up by running
    # jobs without a relaunch. Empty -> the static `retriever_url`.
    retriever_endpoint_file: str = ""
    timeout_sec: float = 600.0
    max_memory_mb: int = 2048
    max_workers: int = 16
    topk_default: int = 3
    max_search_calls: int = 0
    # Cap on candidates sent to the cross-encoder per `rerank()` call; 0 = unlimited.
    rerank_max_docs: int = 0
    # Cap on variables carried between blocks of one rollout.
    max_state_bytes: int = 4 * 1024 * 1024
    max_output_bytes: int = 64 * 1024
    # Relevance probability above which `rerank()` keeps a document when called without `top_k`.
    rerank_relevance_threshold: float = 0.5
    # >0 restores MARAG's old fixed-fraction filter instead of the predicate, for that A/B.
    marag_filter_frac: float = 0.0
    # MARAG's fixed retrieval breadth; 20 is the value their implementation section states.
    marag_topk: int = 20

    def __post_init__(self):
        # SANDBOX_POOL=1 -> reuse warm workers (fast); else fork+exec per block (default, robust).
        self._use_pool = os.environ.get("SANDBOX_POOL") == "1"
        self._idle: "queue.Queue" = queue.Queue()
        self._timing = os.environ.get("SANDBOX_TIMING") == "1"  # per-turn latency split logging
        self._ep_ttl = 30.0  # re-read a published endpoint JSON at most this often (s)
        self._ep_cache = {"retriever": [self.retriever_url, 0.0], "rerank": [self.rerank_url, 0.0]}  # [url, ts]

    def _resolve(self, kind: str, static_url: str, endpoint_file: str) -> str:
        """Resolve a service URL, re-reading its published endpoint JSON at most every `_ep_ttl` s so a
        restarted server (new task IP) is picked up by a running job without a relaunch. Keeps the last good URL
        on a read error (server mid-restart / file mid-write). No endpoint file -> the static url."""
        if not endpoint_file:
            return static_url
        c = self._ep_cache[kind]
        now = time.time()
        if now - c[1] >= self._ep_ttl:
            c[1] = now
            try:
                with open(endpoint_file) as f:
                    u = json.load(f).get("url")
                if u:
                    c[0] = u
            except (OSError, ValueError):
                pass
        return c[0]

    def _env(self) -> dict:
        env = {
            "PATH": os.environ.get("PATH", ""),
            "RETRIEVER_URL": self._resolve("retriever", self.retriever_url, self.retriever_endpoint_file),
            "RERANK_URL": self._resolve("rerank", self.rerank_url, self.rerank_endpoint_file),
            "TOPK_DEFAULT": str(self.topk_default),
            "MAX_SEARCH_CALLS": str(self.max_search_calls),
            "RERANK_MAX_DOCS": str(self.rerank_max_docs),
            "MAX_STATE_BYTES": str(self.max_state_bytes),
            "MAX_OUTPUT_BYTES": str(self.max_output_bytes),
            "RERANK_RELEVANCE_THRESHOLD": str(self.rerank_relevance_threshold),
            "MARAG_FILTER_FRAC": str(self.marag_filter_frac),
            "MARAG_TOPK": str(self.marag_topk),
            "SANDBOX_MEM_MB": str(self.max_memory_mb),  # child self-applies RLIMIT_AS from this
            # Under that RLIMIT_AS, OpenBLAS starting one thread per core hangs `import numpy` (so pandas too)
            # until the wall-clock timeout.
            "OPENBLAS_NUM_THREADS": "1",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            # the retriever is on localhost, so it must bypass any HTTP proxy the host exports.
            # Both spellings deliberately: the SDK's own urllib calls would take either (CPython
            # lowercases env names when it builds the proxy map), but the sandbox executes arbitrary
            # model-generated code, and `requests` / `httpx` follow their own casing conventions.
            "no_proxy": "127.0.0.1,localhost,0.0.0.0",
            "NO_PROXY": "127.0.0.1,localhost,0.0.0.0",
        }
        if self._use_pool:
            env["SANDBOX_POOL"] = "1"
        else:
            # One-shot: bound this block's CPU in the child. A pool worker runs MANY blocks, so RLIMIT_CPU
            # (cumulative) would kill a healthy warm worker -- the parent's per-request timeout guards it.
            env["SANDBOX_CPU_SEC"] = str(int(self.timeout_sec) + 1)
        for k in ("LD_LIBRARY_PATH", "SEARCH_SDK_DIR"):  # LD_LIBRARY_PATH: some Python builds need it
            if k in os.environ:
                env[k] = os.environ[k]
        return env

    # ---- one-shot backend (default) ----
    def _run_oneshot(self, code: str, state: dict | None = None) -> ExecResult:
        proc = subprocess.Popen(
            [sys.executable, "-s", "-B", _RUNNER],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            env=self._env(), start_new_session=True, text=True,  # own session -> killpg; rlimits in child
        )
        try:
            req = json.dumps({"code": code, "state": state or {}})
            out, _err = proc.communicate(input=req, timeout=self.timeout_sec)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except OSError:  # ProcessLookupError (already gone) / PermissionError -- either way, move on
                pass
            try:
                proc.communicate(timeout=5)  # reap; never block forever if the group didn't fully die
            except subprocess.TimeoutExpired:
                pass
            return ExecResult("", None, True)
        try:
            env = json.loads(out)
            return ExecResult(
                env.get("stdout", ""), env.get("error"), False,
                int(env.get("search_calls", 0)),
                dense_calls=int(env.get("dense_calls", 0)),
                sparse_calls=int(env.get("sparse_calls", 0)),
                search_time=float(env.get("search_time", 0.0)),
                rerank_calls=int(env.get("rerank_calls", 0)),
                call_docs=env.get("call_docs") or [],
                rerank_call_docs=env.get("rerank_call_docs") or [],
                call_doc_ids=env.get("call_doc_ids") or [],
                state=env.get("state") or {},
            )
        except (json.JSONDecodeError, ValueError):
            # Child died before emitting the envelope (e.g. OOM-killed by RLIMIT_AS).
            return ExecResult("", f"sandbox produced no result (rc={proc.returncode})", False)

    # ---- pool backend (SANDBOX_POOL=1) ----
    def _spawn(self) -> subprocess.Popen:
        return subprocess.Popen(
            [sys.executable, "-s", "-B", _RUNNER],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            env=self._env(), start_new_session=True, bufsize=0,  # own session -> killpg; rlimits in child
        )

    @staticmethod
    def _readn(stream, n: int) -> bytes:
        buf = bytearray()
        while len(buf) < n:
            chunk = stream.read(n - len(buf))
            if not chunk:
                raise EOFError("worker closed")
            buf += chunk
        return bytes(buf)

    def _exchange(self, w: subprocess.Popen, code: str, state: dict | None = None) -> dict:
        payload = json.dumps({"code": code, "state": state or {}}).encode("utf-8")
        w.stdin.write(f"{len(payload)}\n".encode())
        w.stdin.write(payload)
        w.stdin.flush()
        header = w.stdout.readline()
        if not header:
            raise EOFError("no response header")
        return json.loads(self._readn(w.stdout, int(header)))

    def _run_pooled(self, code: str, state: dict | None = None) -> ExecResult:
        try:
            w = self._idle.get_nowait()
        except queue.Empty:
            w = self._spawn()
        box: dict = {}
        done = threading.Event()

        def _call():
            try:
                box["env"] = self._exchange(w, code, state)
            except Exception as e:  # worker died / framing error (EOFError / OSError / JSON / ValueError)
                box["err"] = e
            finally:
                done.set()

        threading.Thread(target=_call, daemon=True).start()
        if not done.wait(self.timeout_sec):  # hung -> kill the whole worker group, don't reuse it
            try:
                os.killpg(w.pid, signal.SIGKILL)
            except OSError:  # ProcessLookupError / PermissionError -- worker already gone
                pass
            return ExecResult("", None, True)
        if "err" in box:  # dead/broken worker -> drop it (a fresh one spawns on the next request)
            try:
                w.kill()
            except OSError:
                pass
            return ExecResult("", "sandbox produced no result", False)
        self._idle.put(w)  # healthy -> return to the pool
        env = box["env"]
        return ExecResult(
            env.get("stdout", ""), env.get("error"), False,
            int(env.get("search_calls", 0)),
            dense_calls=int(env.get("dense_calls", 0)),
            sparse_calls=int(env.get("sparse_calls", 0)),
            search_time=float(env.get("search_time", 0.0)),
            rerank_calls=int(env.get("rerank_calls", 0)),
            call_docs=env.get("call_docs") or [],
            rerank_call_docs=env.get("rerank_call_docs") or [],
            call_doc_ids=env.get("call_doc_ids") or [],
            state=env.get("state") or {},
        )

    def run_one(self, code: str, state: dict | None = None) -> ExecResult:
        if not code or not code.strip():
            # Preserve the caller's namespace: an empty block is a no-op, not a reason to lose state.
            return ExecResult("", "empty code block", False, state=dict(state or {}))
        t0 = time.perf_counter()
        r = self._run_pooled(code, state) if self._use_pool else self._run_oneshot(code, state)
        r.run_time = time.perf_counter() - t0
        return r

    def run_batch_detailed(self, codes: list, states: list | None = None) -> list:
        """Run one block per entry. `states` (optional, same length) seeds each block's namespace."""
        t0 = time.perf_counter()
        states = states or [None] * len(codes)
        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            results = list(pool.map(self.run_one, codes, states))
        if self._timing and results:  # SANDBOX_TIMING=1: per-turn split of retrieval vs the rest
            wall = time.perf_counter() - t0
            sum_run = sum(r.run_time for r in results)
            sum_search = sum(r.search_time for r in results)
            print(f"SANDBOX_TIMING: blocks={len(results)} wall={wall:.2f}s sum_block={sum_run:.2f}s "
                  f"sum_retrieval={sum_search:.2f}s retrieval={100 * sum_search / max(sum_run, 1e-9):.0f}%_of_block",
                  flush=True)
        return results

    def run_batch(self, codes: list) -> list:
        return [r.observation() for r in self.run_batch_detailed(codes)]
