# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""The dropped-state warning must survive every truncation between the sandbox and the model.

Three caps interact. `MAX_STATE_BYTES` decides whether the namespace carries into the next block;
`MAX_OUTPUT_BYTES` truncates the sandbox's stdout; then `max_obs_length` truncates again when the agent
loop tokenizes the observation. Both truncations keep the HEAD, and a block big enough to blow the state
cap has usually blown the output caps too -- so appending the warning put it exactly where both cuts land.
The state then vanished silently and the next block raised `NameError` on its own variable. Hence the
warning is PREPENDED, and these tests pin that: an append-at-the-end fix passes the first assertion below
and still loses the warning downstream at `max_obs_length`.

That is not a cosmetic loss. One run spent 150 steps with `NameError` in 99.9% of rollouts and its retrieval
metric frozen at roughly two thirds of what a comparable arm reached once its state fit, while the dumps
showed the warning in ~0% of rollouts -- so it read as "the model cannot chain tool calls" rather than as a
namespace the harness threw away.

Only stdlib imports, so this runs without torch/verl. The SDK is passed as a stub: these blocks never
retrieve.
"""

import unittest

from rpl.llm_agent import sandbox_runner as sr


class _Stub:
    """Stands in for the `search_sdk` module. `_run_one` reads counters off it and injects its primitives."""

    def __getattr__(self, name):
        return 0


class StateWarningSurvivesTruncationTest(unittest.TestCase):
    def setUp(self):
        self._caps = (sr._MAX_STATE_BYTES, sr._MAX_OUTPUT_BYTES)
        # Small enough that one print blows both, which is the GlobalQA case in miniature: there a single
        # `print(docs)` of ~100 full resumes is hundreds of KB against a 64 KB output cap.
        sr._MAX_STATE_BYTES, sr._MAX_OUTPUT_BYTES = 100, 200

    def tearDown(self):
        sr._MAX_STATE_BYTES, sr._MAX_OUTPUT_BYTES = self._caps

    def _run(self, code):
        return sr._run_one(code, _Stub())

    def test_the_warning_is_present_when_a_big_print_also_blows_the_output_cap(self):
        env = self._run('x = "A" * 4000\nprint(x)')
        self.assertEqual(env["state"], {}, "state should have been dropped")
        self.assertIn("state exceeded", env["stdout"],
                      "the warning was truncated away, so the model cannot diagnose the NameError it is "
                      "about to hit on `x`")

    def test_the_document_body_is_still_truncated(self):
        """The fix must not smuggle the whole output past the cap -- only the short warning rides along."""
        env = self._run('x = "A" * 4000\nprint(x)')
        self.assertLess(len(env["stdout"]), 200 + 200, "output body is no longer being truncated")
        self.assertNotIn("A" * 1000, env["stdout"])

    def test_a_small_block_keeps_its_state_and_says_nothing(self):
        env = self._run('y = 1\nprint(y)')
        self.assertEqual(env["state"].get("y"), 1)
        self.assertNotIn("state exceeded", env["stdout"])

    def test_the_warning_survives_a_downstream_head_keeping_truncation(self):
        """`max_obs_length` cuts the observation's tail (rpl_agent_loop.py, the `obs_ids[:n] + tail`
        slice). So being inside the stdout is not enough -- the warning has to be near the FRONT."""
        env = self._run('x = "A" * 4000\nprint(x)')
        head = env["stdout"][:120]  # a small head, as a stand-in for a tight max_obs_length
        self.assertIn("state exceeded", head,
                      "the warning is not in the head, so max_obs_length will still cut it")

    def test_the_warning_survives_even_with_no_stdout_at_all(self):
        """A block that assigns a huge value without printing must still be told the state was dropped."""
        env = self._run('x = "A" * 4000')
        self.assertEqual(env["state"], {})
        self.assertIn("state exceeded", env["stdout"])


if __name__ == "__main__":
    unittest.main()
