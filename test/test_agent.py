#! /usr/bin/env python3
# coding=utf-8

# Ruibo Liu @Dartmouth College
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for Agent."""

from absl import logging
from absl.testing import absltest, parameterized

from stable_alignment.sandbox import Agent

logging.set_verbosity("info")
logging.set_stderrthreshold("info")


class AgentTest(parameterized.TestCase):
    """Test cases for Social Agents."""

    def test_agent_internal_memory(self):
        """Test the save/update functions of agent's internal memory."""
        agent = Agent(agent_id=19, location=(3, 4), world_id=0, label="good")
        self.assertEqual(agent.model_type, "text-davinci-002")
        agent.reset_memory()
        agent.save_int_memory(
            question="What's the weather like today?", answer="It is pretty good!"
        )
        self.assertDictEqual(
            agent.internal_mem, {"What's the weather like today?": "It is pretty good!"}
        )

    def test_agent_response(self):
        """Test whether the agent is able to generate self-consistent answers."""
        agent = Agent(
            agent_id=0,
            location=(0, 0),
            model_type="text-davinci-003",
            world_id=0,
            label="good",
        )
        self.assertEqual(agent.model_type, "text-davinci-003")
        agent.reset_memory()
        agent.save_int_memory(
            question="Do you love any ball games?",
            answer="I love all of them except basketball!",
        )
        logging.info(
            agent.response("Do you want to play basketball with me?", verbose=True)
        )

    def test_agent_response_chat_gpt(self):
        """Test whether the agent is able to generate answers with GPT-3.5 engine."""
        agent = Agent(
            agent_id=0,
            location=(0, 0),
            model_type="gpt-3.5-turbo",
            world_id=0,
            label="good",
        )
        self.assertEqual(agent.model_type, "gpt-3.5-turbo")
        agent.reset_memory()
        agent.save_int_memory(
            question="Do you love any ball games?",
            answer="I love all of them except basketball!",
        )
        logging.info(
            agent.response("Do you want to play basketball with me?", verbose=True)
        )

    def test_agent_init_with_paths_no_world_id(self):
        """Test that we can initialize the agent with only memory and embedding paths."""
        agent = Agent(
            agent_id=1,
            location=(0, 1),
            int_mem_path="./data/cache/world_0/internal_memory/agent_1.pkl",
            int_mem_emb_path="./data/cache/world_0/internal_memory/agent_1_emb.pkl",
            ext_mem_path="./data/cache/world_0/external_memory/agent_1.jsonl",
            label="good",
        )
        agent.reset_memory()
        agent.save_int_memory(
            question="Do you love any ball games?",
            answer="I love all of them except basketball!",
        )
        logging.info(
            agent.response("Do you want to play basketball with me?", verbose=True)
        )

    def test_agent_init_with_paths_expect_fail(self):
        """Test that initializing the agent with no world id and not all three paths would assert false."""
        try:
            Agent(
                agent_id=1,
                location=(0, 1),
                model_type="text-davinci-002",
                int_mem_path="./data/cache/world_0/internal_memory/agent_1.pkl",
                int_mem_emb_path="./data/cache/world_0/internal_memory/agent_1_emb.pkl",
                label="good",
            )
        except AssertionError as e:
            logging.info(str(e))
        else:
            self.fail("Should raise an AssertionError.")


if __name__ == "__main__":
    absltest.main()
