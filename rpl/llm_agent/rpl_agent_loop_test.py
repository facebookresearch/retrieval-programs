# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for the agent loop's observation cap, on live observations and on a program-mutation `response_prefix`.

A character-level tokenizer stands in for the real one, so a token budget is a character budget.
"""

import unittest

from rpl.llm_agent.rpl_agent_loop import RPLAgentLoop


class CharTokenizer:
    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [ord(c) for c in text]}


def loop_with_cap(max_obs_length: int) -> RPLAgentLoop:
    loop = object.__new__(RPLAgentLoop)
    loop.tokenizer = CharTokenizer()
    loop.max_obs_length = max_obs_length
    return loop


def text(ids: list[int]) -> str:
    return "".join(map(chr, ids))


TURN = "<code>\nprint(dense_retrieval('q', top_k=20))\n</code>"


def observation(body: str, tag: str = "output") -> str:
    return f"\n\n<{tag}>{body}</{tag}>\n\n"


class CapObservationTest(unittest.TestCase):
    def test_a_long_observation_is_cut_to_the_budget_and_stays_terminated(self):
        obs = observation("x" * 200)
        capped = text(loop_with_cap(60)._cap_observation(obs, [ord(c) for c in obs]))
        self.assertEqual(len(capped), 60)
        self.assertTrue(capped.endswith("</output>\n\n"))
        self.assertTrue(obs.startswith(capped[: -len("</output>\n\n")]))

    def test_an_information_block_keeps_its_own_tag(self):
        obs = observation("x" * 200, "information")
        self.assertTrue(text(loop_with_cap(60)._cap_observation(obs, [ord(c) for c in obs])).endswith(
            "</information>\n\n"))

    def test_a_short_observation_and_a_zero_budget_are_untouched(self):
        obs = observation("x" * 200)
        ids = [ord(c) for c in obs]
        self.assertEqual(loop_with_cap(len(ids))._cap_observation(obs, ids), ids)
        self.assertEqual(loop_with_cap(0)._cap_observation(obs, ids), ids)


class PrefixIdsTest(unittest.TestCase):
    """The verify / control prefix ends with an observation fabricated offline, which only the loop can cap."""

    def test_the_final_observation_is_capped_and_the_turns_before_it_are_kept(self):
        head = "I search first." + TURN
        prefix = head + observation("d" * 500)
        got = text(loop_with_cap(100)._prefix_ids(prefix))
        self.assertTrue(got.startswith(head + "\n\n<output>"))
        self.assertEqual(len(got) - len(head), 100)
        self.assertTrue(got.endswith("</output>\n\n"))

    def test_it_is_capped_exactly_as_a_live_observation_would_be(self):
        loop, obs = loop_with_cap(100), observation("d" * 500)
        live = loop._cap_observation(obs, loop._tokenize(obs))
        self.assertEqual(loop._prefix_ids(TURN + obs), loop._tokenize(TURN) + live)

    def test_a_prefix_within_budget_is_tokenized_whole_as_before(self):
        prefix = TURN + observation("d" * 50)
        self.assertEqual(loop_with_cap(100)._prefix_ids(prefix), [ord(c) for c in prefix])

    def test_only_the_final_observation_is_capped(self):
        # An earlier observation was capped when the rollout ran; re-cutting it would change the policy's own turn.
        prefix = TURN + observation("e" * 500) + TURN + observation("d" * 50)
        self.assertEqual(loop_with_cap(100)._prefix_ids(prefix), [ord(c) for c in prefix])

    def test_a_prefix_that_does_not_end_with_an_observation_is_untouched(self):
        prefix = TURN + observation("e" * 500) + "Now I answer."
        self.assertEqual(loop_with_cap(100)._prefix_ids(prefix), [ord(c) for c in prefix])

    def test_no_budget_means_no_cap(self):
        prefix = TURN + observation("d" * 500)
        self.assertEqual(loop_with_cap(0)._prefix_ids(prefix), [ord(c) for c in prefix])


if __name__ == "__main__":
    unittest.main()
