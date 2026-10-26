# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Child entrypoint for the subprocess sandbox.

Two modes (selected by SANDBOX_POOL):
  - one-shot (default): read model Python from stdin, exec it, write ONE JSON envelope, exit. Simple;
    a fresh process (fork+exec+import) per code block.
  - pool (SANDBOX_POOL=1): import search_sdk once, then serve a stream of code blocks over length-framed
    stdin/stdout so the parent can reuse this warm worker (no fork+exec+import per block). Protocol:
        parent -> child:  b"<nbytes>\n" + <code utf-8>
        child  -> parent: b"<nbytes>\n" + <json envelope utf-8>
    EOF on stdin -> exit.

The envelope is {"stdout": <captured>, "error": <last traceback line or null>, "search_calls": <int>}.
"""
from __future__ import annotations

import contextlib
import io
import ipaddress
import json
import os
import resource
import socket
import sys
import traceback
import urllib.parse

_MAX_OUTPUT_BYTES = int(os.environ.get("MAX_OUTPUT_BYTES", str(64 * 1024)))
# Cap on the variables carried between blocks of one rollout.
_MAX_STATE_BYTES = int(os.environ.get("MAX_STATE_BYTES", str(4 * 1024 * 1024)))


def _capture_state(namespace: dict, injected: set) -> tuple[dict, str | None]:
    """Return the JSON-serializable user variables to carry into the next block of this rollout.

    Only plain data survives: the SDK primitives are re-injected per block, and anything unserializable
    (a generator, a module, a closure) is dropped instead of failing the block.
    """
    out = {}
    for k, v in namespace.items():
        if k in injected or k.startswith("__"):
            continue
        try:
            json.dumps(v)
        except (TypeError, ValueError):
            continue
        out[k] = v
    if not out:
        return {}, None
    if len(json.dumps(out)) > _MAX_STATE_BYTES:
        return {}, f"sandbox state exceeded {_MAX_STATE_BYTES} bytes and was dropped; recompute in this block"
    return out, None


def _install_sdk_on_path() -> None:
    sdk_dir = os.environ.get("SEARCH_SDK_DIR")
    if not sdk_dir:
        # search_sdk.py lives at ../search relative to this file.
        here = os.path.dirname(os.path.abspath(__file__))
        sdk_dir = os.path.join(os.path.dirname(here), "search")
    if sdk_dir not in sys.path:
        sys.path.insert(0, sdk_dir)


def _fmt_search(res):
    if not isinstance(res, list):
        return str(res)
    return "\n".join(
        f"Doc {i+1}(Title: {d.get('title','')}) {d.get('text','')}" if isinstance(d, dict) else str(d)
        for i, d in enumerate(res)
    )


def _run_one(code: str, search_sdk, state: dict | None = None) -> dict:
    """Execute one code block and return the result envelope dict.

    `state` is the caller-held namespace from the previous block of the same rollout. The parent owns it,
    so there is no worker affinity to maintain: it travels with the rollout and dies with it. Without this,
    `docs = dense_retrieval(...)` / `rerank(q, docs, 3)` split across two turns is a guaranteed NameError.

    Auto-echo: models often call search(...) without print(), so results never reach stdout and the
    observation comes back empty (the model then answers from memory). Record each search()'s results
    and, if the code printed nothing, surface them. If the code did print, use its stdout as-is.
    """
    search_sdk._search_calls = 0          # reset per block (the pool worker reuses the module)
    search_sdk._dense_calls = 0           # per-method counts (complementary-search reward + routing analytics)
    search_sdk._sparse_calls = 0
    search_sdk._rerank_calls = 0
    search_sdk._search_time = 0.0         # reset per block (retriever HTTP time, latency profiling)
    search_sdk._call_docs = []            # per-call retrieved doc texts (shipped out for answer recall/precision)
    search_sdk._rerank_call_docs = []     # per-call reranked doc texts (rerank precision diagnostic)
    search_sdk._call_doc_ids = []         # per-call retrieved corpus ids (GlobalQA document-coverage F1)
    auto_echo = []

    def _wrap(fn):
        # Wrap an SDK retrieval primitive so a bare call (no print) still auto-echoes its results.
        # All primitives share search_sdk's call counter/timer.
        def _wrapped(*args, **kwargs):
            res = fn(*args, **kwargs)
            try:
                auto_echo.append(_fmt_search(res))
            except Exception:
                pass
            return res
        return _wrapped

    primitives = {"search": _wrap(search_sdk.search)}
    # Multi-retriever primitives route to the merged server's backends (dense=e5/faiss, sparse=bm25);
    # inject whichever the SDK exposes.
    for _name in ("dense_retrieval", "sparse_retrieval"):
        if hasattr(search_sdk, _name):
            primitives[_name] = _wrap(getattr(search_sdk, _name))
    # fuse/dedup are pure post-processing transforms; rerank calls the reranker service but likewise fetches
    # no new docs.
    for _name in ("fuse", "dedup", "rerank"):
        if hasattr(search_sdk, _name):
            primitives[_name] = getattr(search_sdk, _name)
    # MARAG-R1 tool surface (`marag` mode): fixed-configuration wrappers, injected the same way so a missing
    # one is a NameError at the call site rather than a silently absent capability. DenseSearch/KeywordSearch
    # auto-echo like the other retrievals.
    for _name in ("DenseSearch", "KeywordSearch"):
        if hasattr(search_sdk, _name):
            primitives[_name] = _wrap(getattr(search_sdk, _name))
    if hasattr(search_sdk, "DocumentFilter"):
        primitives["DocumentFilter"] = search_sdk.DocumentFilter
    if hasattr(search_sdk, "get_document"):
        primitives["get_document"] = search_sdk.get_document
    # Restored state first, primitives second: a model that assigns to `rerank` loses its variable rather
    # than shadowing the primitive, which would surface as a bewildering "'list' object is not callable".
    namespace = {"__name__": "__main__", **(state or {}), **primitives}
    injected = set(primitives) | {"__name__"}

    out_buf, err_buf = io.StringIO(), io.StringIO()
    error = None
    try:
        with contextlib.redirect_stdout(out_buf), contextlib.redirect_stderr(err_buf):
            exec(compile(code, "<model_code>", "exec"), namespace)
    except BaseException:
        tb = traceback.format_exc().rstrip().splitlines()
        error = tb[-1].strip() if tb else "error"

    next_state, state_warn = _capture_state(namespace, injected)
    stdout = out_buf.getvalue()
    if not stdout.strip() and auto_echo:  # searched but printed nothing -> auto-echo results
        stdout = "\n\n".join(auto_echo)
    if err_buf.getvalue():
        stdout = (stdout + "\n" + err_buf.getvalue()) if stdout else err_buf.getvalue()
    # PREPENDED, not appended: a block that prints enough to blow the state cap has usually also blown the
    # output caps, and BOTH of them keep the head and drop the tail -- `_MAX_OUTPUT_BYTES` here, then
    # `max_obs_length` when the agent loop tokenizes the observation. Appending put the one line explaining
    # the drop exactly where both truncations cut.
    if state_warn:  # surface the drop to the model; a silent one becomes a NameError it cannot diagnose
        stdout = (state_warn + "\n" + stdout) if stdout else state_warn
    stdout = stdout.encode("utf-8", "replace")[:_MAX_OUTPUT_BYTES].decode("utf-8", "replace")
    return {
        "stdout": stdout, "error": error, "state": next_state,
        "search_calls": int(getattr(search_sdk, "_search_calls", 0)),
        "dense_calls": int(getattr(search_sdk, "_dense_calls", 0)),
        "sparse_calls": int(getattr(search_sdk, "_sparse_calls", 0)),
        "rerank_calls": int(getattr(search_sdk, "_rerank_calls", 0)),
        "search_time": float(getattr(search_sdk, "_search_time", 0.0)),
        "call_docs": getattr(search_sdk, "_call_docs", []),
        "rerank_call_docs": getattr(search_sdk, "_rerank_call_docs", []),
        "call_doc_ids": getattr(search_sdk, "_call_doc_ids", []),
    }


def _readn(stream, n: int):
    """Read exactly n bytes from a binary stream; None on EOF."""
    buf = bytearray()
    while len(buf) < n:
        chunk = stream.read(n - len(buf))
        if not chunk:
            return None
        buf += chunk
    return bytes(buf)


def _decode_req(raw: str) -> tuple[str, dict]:
    """Parse a request frame `{"code": ..., "state": {...}}`.

    Strict on purpose: parent and child ship in the same tar, so there is no version skew to tolerate, and
    accepting a bare code string as a fallback would misparse the blocks that happen to be valid JSON
    (`[1, 2]` is both). A malformed frame becomes a visible error, not a silently empty namespace.
    """
    obj = json.loads(raw)
    if not isinstance(obj, dict) or "code" not in obj:
        raise ValueError("sandbox request must be a JSON object with a 'code' key")
    return obj["code"] or "", obj.get("state") or {}


def _serve(search_sdk) -> None:
    """Pool mode: handle length-framed request frames until the parent closes stdin."""
    inp, outp = sys.stdin.buffer, sys.stdout.buffer
    while True:
        header = inp.readline()
        if not header:
            break  # parent closed stdin -> exit
        req_bytes = _readn(inp, int(header))
        if req_bytes is None:
            break
        code, state = _decode_req(req_bytes.decode("utf-8", "replace"))
        payload = json.dumps(_run_one(code, search_sdk, state)).encode("utf-8")
        outp.write(f"{len(payload)}\n".encode())
        outp.write(payload)
        outp.flush()


def _oneshot(search_sdk) -> None:
    """One-shot mode: one request frame from stdin -> one JSON envelope on stdout."""
    code, state = _decode_req(sys.stdin.read())
    env = _run_one(code, search_sdk, state)
    sys.stdout.write(json.dumps(env))
    sys.stdout.flush()


def _set_rlimits() -> None:
    """Apply OS resource limits to THIS worker, from env set by the parent.

    Done here in the child rather than via a Popen preexec_fn, which is unsafe with the parent's thread
    pool (a fork while another thread holds a lock can deadlock the child before exec). RLIMIT_CPU is
    applied only for one-shot (SANDBOX_CPU_SEC): a pool worker runs many blocks, so the parent's
    per-request timeout bounds hangs instead (RLIMIT_CPU is cumulative -- it would kill a warm worker).
    """
    mem_mb = int(os.environ.get("SANDBOX_MEM_MB", "0"))
    if mem_mb:
        mem = mem_mb * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (mem, mem))
    resource.setrlimit(resource.RLIMIT_FSIZE, (0, 0))  # no file writes (stdout is a pipe, unaffected)
    cpu_sec = int(os.environ.get("SANDBOX_CPU_SEC", "0"))
    if cpu_sec:
        resource.setrlimit(resource.RLIMIT_CPU, (cpu_sec, cpu_sec))


def _restrict_network() -> None:
    """Confine the sandbox to the retriever endpoint; block all other network from model code.

    Model code can `import requests; requests.get(url)` -- external calls time out (slowing the rollout)
    and are an egress hole. Patch socket `getaddrinfo`/`connect`/`connect_ex` to allow only loopback and
    the configured `RETRIEVER_URL` host, refusing everything else fast.
    """
    _orig_getaddrinfo = socket.getaddrinfo
    allow = {"127.0.0.1", "localhost", "::1", "0.0.0.0"}
    # Allow the retriever + reranker wherever they live.
    for _svc_url in (os.environ.get("RETRIEVER_URL", "http://127.0.0.1:8000"), os.environ.get("RERANK_URL", "")):
        svc_host = urllib.parse.urlparse(_svc_url).hostname if _svc_url else None
        if not svc_host:
            continue
        allow.add(svc_host)
        try:
            allow.update(info[4][0] for info in _orig_getaddrinfo(svc_host, None))
        except OSError:
            pass

    def _allowed(host) -> bool:
        if host in allow:
            return True
        try:
            return ipaddress.ip_address(host).is_loopback
        except ValueError:
            return False

    def _guarded_getaddrinfo(host, *args, **kwargs):
        if not _allowed(host):
            raise OSError(f"sandbox: external network blocked (host {host!r})")
        return _orig_getaddrinfo(host, *args, **kwargs)

    def _guard(orig):
        def _connect(self, address):
            if isinstance(address, (tuple, list)) and address and not _allowed(address[0]):
                raise OSError(f"sandbox: external network blocked (host {address[0]!r})")
            return orig(self, address)

        return _connect

    socket.getaddrinfo = _guarded_getaddrinfo
    socket.socket.connect = _guard(socket.socket.connect)
    socket.socket.connect_ex = _guard(socket.socket.connect_ex)


def main() -> None:
    _set_rlimits()
    _install_sdk_on_path()
    import search_sdk  # after sys.path setup; imported once per worker lifetime (pool) or per block (one-shot)

    # Network isolation: model code may issue external HTTP (requests.get to arbitrary URLs) -- those
    # time out (slowing the rollout) and are an egress hole. Confine to loopback + the retriever unless
    # explicitly disabled. After the SDK import (sets up its session), before any model code runs.
    if os.environ.get("SANDBOX_ALLOW_NET") != "1":
        _restrict_network()

    if os.environ.get("SANDBOX_POOL") == "1":
        _serve(search_sdk)
    else:
        _oneshot(search_sdk)


if __name__ == "__main__":
    main()
