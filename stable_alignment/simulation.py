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
"""Launcher script for sandbox simulation.

Example usage:

python run_simulation.py -model_type 'text-davinci-003' -obs_model_type 'gpt-3.5-turbo' -world_id 1 -init_setting 'all_bad' -n_round '4' -size '10' -dataset_name 'hh-rlhf'
"""

import argparse
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, List

import openai
from absl import logging
from tqdm import tqdm

from stable_alignment.sandbox import Agent, World, get_query_questions, load_initial_data

logging.set_verbosity("info")
logging.set_stderrthreshold("info")


def one_agent_one_iteration(
    question: str, agent: Agent, world: World, iteration: int
) -> str:
    """Single thread version of interaction_single_round."""
    draft_ans = agent.response(question, verbose=False)
    message = world.back_scatter(
        iteration,
        agent,
        question,
        draft_ans,
        dropout_rate=0.5,
        tgt_agent_count=4,
    )
    return message


def many_agents_one_iteration(
    questions: List[str], agents: List[Agent], world: World, iteration: int
) -> None:
    """Multi thread version of interaction_single_round."""
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [
            executor.submit(one_agent_one_iteration, question, agent, world, iteration)
            for question, agent in zip(questions, agents)
        ]
        for future in as_completed(futures):
            logging.info(future.result())


def interaction_single_round(world: World, iteration: int, single_thread: bool) -> None:
    """
    Simulate a single round of interation of a world state, updating relevant memory.
    Current approach: iterate through each participant (active agent) in the world,
                      - for each, perform the following two stages:
                         - draft answer
                         - back-scatter for final answer.
    Should update the world state
    (participants' internal and external memory, and subsequently their moral scores).
    """
    questions = get_query_questions(args.dataset_name, len(world.participants), iteration)
    if single_thread:
        for idx, agent in enumerate(world.participants):
            question = questions[idx]
            draft_ans = agent.response(question, verbose=False)
            world.back_scatter(
                iteration,
                agent,
                question,
                draft_ans,
                dropout_rate=0.8,
                tgt_agent_count=16,
            )
    else:
        many_agents_one_iteration(questions, world.participants, world, iteration)


def main(args: Any) -> None:
    openai.api_key_path = args.api_key_path
    openai.api_key = os.getenv("OPENAI_API_KEY")

    world = World(
        world_id=args.world_id,
        grid_size=args.size,
        initial_setting=args.init_setting,
        local_interaction=args.local_interaction,
        global_interaction=args.global_interaction,
        model_type=args.model_type,
        obs_model_type=args.obs_model_type,
        score_only=False,
        has_prior_mem=True,
        initial_data=load_initial_data(args.dataset_name),
        dataset_name=args.dataset_name,
        verbose=True,
    )
    for i in tqdm(range(args.n_round)):
        interaction_single_round(world, i, args.single_thread)
        # time.sleep(60)


# writer reader

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-api_key_path",
        default=".env",
        type=str,
        help="path to the env file with openai key",
    )
    parser.add_argument(
        "-model_type",
        default="text-davinci-002",
        choices=[
            "code‑davinci‑002",
            "text-davinci-002",
            "text-davinci-003",
            "text-davinci-001",
            "text-curie-001",
            "text-babbage-001",
            "text-ada-001",
            "gpt-3.5-turbo",
            "gpt-3.5-turbo-0301",
            "gpt-4",
            "gpt-4-0314",
        ],  # GPT-3, 3.5 & 4
        type=str,
        help="model type of the agents",
    )
    parser.add_argument(
        "-obs_model_type",
        default="text-davinci-003",
        choices=[
            "text-davinci-003",
            "gpt-3.5-turbo",
        ],  # GPT-3, 3.5 & 4
        type=str,
        help="model type of the observers",
    )
    parser.add_argument(
        "-score_only",
        default=False,
        type=bool,
        help="whether the feedback only takes scores",
    )
    parser.add_argument(
        "-single_thread",
        default=False,
        type=bool,
        help="whether the simulation runs in a single thread",
    )
    parser.add_argument(
        "-n_round", default=1, type=int, help="number of rounds of interaction"
    )
    parser.add_argument("-world_id", type=int, help="world id")
    parser.add_argument("-size", default=3, type=int, help="size of the grid")
    parser.add_argument(
        "-init_setting",
        choices=["all_good", "all_bad", "half_half", "mixed_half_half"],
        type=str,
        help="initial demographics setting",
    )
    parser.add_argument(
        "-local_interaction",
        default=True,
        type=bool,
        help="whether the world has local interaction",
    )
    parser.add_argument(
        "-global_interaction",
        default=False,
        type=bool,
        help="whether the world has global/social-media interaction",
    )
    parser.add_argument(
        "-dataset_name",
        default="hh-rlhf",
        choices=["hh-rlhf"],
        type=str,
        help=(
            "name of the dataset for initializing agent's world view"
            "and incoming questions"
        ),
    )
    args = parser.parse_args()
    main(args)
