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
"""Agent Class."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import openai
import pandas as pd
from absl import logging
from openai.embeddings_utils import distances_from_embeddings, get_embedding
from readerwriterlock import rwlock

from stable_alignment.sandbox.utils import call_gpt

EMBED_ENG: str = "text-embedding-ada-002"
CACHE_DIR_PREFIX: str = "./data/cache"
DISTANCE_THRESHOLD = 0.3

openai.api_key_path = "./.env"


class Agent:
    """Class for simulating humans."""

    def __init__(
        self,
        agent_id: int,
        label: str,
        location: Tuple[int, int],
        world_id: int = None,
        model_type: str = "text-davinci-002",
        initial_label: str = "good",
        social_circle_radius: int = 1,
        int_mem_path: str = None,
        int_mem_emb_path: str = None,
        ext_mem_path: str = None,
        initial_mem: dict = None,
        is_active: bool = True,
        embedding_engine: str = EMBED_ENG,
    ):
        """Agent initialization.

        Args:
            agent_id: The unique id of the agent (this is a required arg).
            location: A tuple of int which specifies the location of the agent
                on the grid (0-based index, required arg).
            world_id: The unique id for different world settings.
            model_type: The model type behind the agent.
            initial_label: The initial moral label of the agent,
                can be either "good" or "bad"
            social_circle_radius: The radius of the local interaction.
            int_mem_path: The unique internal memory path of the agent.
            int_mem_emb_path: The unique internal memory embedding path of the agent.
            ext_mem_path: The unique external memory path of the agent.
            initial_mem: The initial internal memory, which is {Q: A} pairs.
            is_active: Whether or not it is active.
            embedding_engine: The embedding engine used to embed internal memory.
        """
        self.agent_id = agent_id
        self.label = label
        self.initial_label = initial_label
        self.social_circle_radius = social_circle_radius
        self.location = location
        self.is_active = is_active
        self.model_type = model_type
        self.embedding_engine = embedding_engine

        self.internal_mem: Dict[str, str] = {}
        self.internal_mem_emb: Dict[Tuple[str, str], Any] = {}

        self.locker = rwlock.RWLockFair()

        # Handling memories
        if world_id is None:
            assert int_mem_path and int_mem_emb_path and ext_mem_path, (
                "No id provided. "
                "You should specify the internal and external memory paths, "
                "and internal memory embedding path."
            )
            self.int_mem_path: str = int_mem_path
            self.int_mem_emb_path: str = int_mem_emb_path
            self.ext_mem_path: str = ext_mem_path
        else:
            self.world_id = world_id
            self.int_mem_path = (
                CACHE_DIR_PREFIX +
                f"/world_{world_id}/internal_memory/agent_{agent_id}.pkl"
            )
            self.int_mem_emb_path = (
                CACHE_DIR_PREFIX +
                f"/world_{world_id}/internal_memory/agent_{agent_id}_emb.pkl"
            )
            self.ext_mem_path = (
                CACHE_DIR_PREFIX +
                f"/world_{world_id}/external_memory/agent_{agent_id}.jsonl"
            )

            Path("/".join(self.int_mem_path.split("/")[:-1])
                 ).mkdir(parents=True, exist_ok=True)
            Path("/".join(self.int_mem_emb_path.split("/")[:-1])
                 ).mkdir(parents=True, exist_ok=True)
            Path("/".join(self.ext_mem_path.split("/")[:-1])
                 ).mkdir(parents=True, exist_ok=True)

            self.reset_memory()

        # Load internal memory and external memory given their paths.
        self._load_int_memory(initial_mem)
        if Path(self.ext_mem_path).is_file():
            read_marker = self.locker.gen_rlock()
            read_marker.acquire()
            # self.external_mem = pd.read_pickle(self.ext_mem_path)
            self.external_mem = pd.read_json(self.ext_mem_path, orient="records")
            read_marker.release()
        else:
            self.external_mem = pd.DataFrame()

    def __repr__(self):  # type: ignore
        return f"Agent ID: {self.agent_id}"

    def save_int_memory(self, question: str, answer: str) -> None:
        """Update q-a pairs (self.internal_mem) and
        the question embeddings (self.internal_mem_emb)."""
        if (question, self.embedding_engine) not in self.internal_mem_emb.keys():
            self.internal_mem_emb.update(
                {(question.strip(), self.embedding_engine): self.get_embedding(question)}
            )

        if question not in self.internal_mem.keys():
            self.internal_mem.update({question: answer})

        write_marker = self.locker.gen_wlock()
        write_marker.acquire()
        pd.to_pickle(self.internal_mem, self.int_mem_path)
        pd.to_pickle(self.internal_mem_emb, self.int_mem_emb_path)
        write_marker.release()

    def save_ext_memory(
        self,
        question: str,
        draft_answer: str,
        iteration: int,
        ratings: List[int],
        tgt_agent_ids: List[int],
        feedbacks: List[str],
        revised_answer: str,
        gen_moral_score_after: float,
        gen_moral_score_before: float,
        gen_moral_reason_after: str,
        gen_moral_reason_before: str,
        gen_engagement_score_after: float,
        gen_engagement_score_before: float,
        gen_engagement_reason_after: str,
        gen_engagement_reason_before: str,
        cls_moral_score_after: float,
        cls_moral_score_before: float,
    ) -> None:
        """Update the records dict for other agents' feedbacks."""

        temp_df = pd.DataFrame.from_records(
            [
                {
                    "agent_id": self.agent_id,
                    "label": self.label,
                    "iteration": iteration,
                    "question": question,
                    "draft_answer": draft_answer,
                    "target_id": tgt_agent_ids,
                    "feedback": feedbacks,
                    "rating": ratings,
                    "revised_answer": revised_answer,
                    "gen_moral_score_before": gen_moral_score_before,
                    "gen_moral_score_after": gen_moral_score_after,
                    "gen_moral_reason_before": gen_moral_reason_before,
                    "gen_moral_reason_after": gen_moral_reason_after,
                    "gen_engagement_score_before": gen_engagement_score_before,
                    "gen_engagement_score_after": gen_engagement_score_after,
                    "gen_engagement_reason_before": gen_engagement_reason_before,
                    "gen_engagement_reason_after": gen_engagement_reason_after,
                    "cls_moral_score_before": cls_moral_score_before,
                    "cls_moral_score_after": cls_moral_score_after,
                }
            ]
        )

        self.external_mem = pd.concat([self.external_mem, temp_df], ignore_index=True)

        write_marker = self.locker.gen_wlock()
        write_marker.acquire()
        # pd.to_pickle(self.external_mem, self.ext_mem_path)
        self.external_mem.to_json(self.ext_mem_path, orient='records', indent=2)
        write_marker.release()

    def _load_int_memory(self, init_data: dict = None) -> None:
        """Load the internal memory given the memory path.

        Args:
            init_data: New data ({Q: A} pairs) to be loaded.

        Note:
            Internal memory has two dicts:
                {Q: A}, for matching answers given questions
                {(Q: embed_engine): Q's embedding}, for searching questions

        Returns:
            The loaded {Q: A} pairs (internal memory), which includes new data.
        """
        if Path(self.int_mem_path).is_file():
            read_marker = self.locker.gen_rlock()
            read_marker.acquire()
            self.internal_mem = pd.read_pickle(self.int_mem_path)
            read_marker.release()
        else:
            self.internal_mem = {}

        if Path(self.int_mem_emb_path).is_file():
            read_marker = self.locker.gen_rlock()
            read_marker.acquire()
            self.internal_mem_emb = pd.read_pickle(self.int_mem_emb_path)
            read_marker.release()
        else:
            self.internal_mem_emb = {}

        if init_data:
            self.internal_mem.update(init_data)

            with ThreadPoolExecutor() as executor:
                res = executor.map(self._save_one_record_init_mem, init_data.keys())

            if len(list(res)) == len(init_data.keys()):
                write_marker = self.locker.gen_wlock()
                write_marker.acquire()
                pd.to_pickle(self.internal_mem, self.int_mem_path)
                pd.to_pickle(self.internal_mem_emb, self.int_mem_emb_path)
                write_marker.release()
            else:
                raise RuntimeError(
                    "Failed to save initial memory. "
                    "Incorrect number of records."
                )

    def _save_one_record_init_mem(self, question: str) -> str:
        """Save one record to the initial memory."""
        self.internal_mem_emb.update(
            {(question.strip(), self.embedding_engine): self.get_embedding(question)}
        )
        return question

    def reset_memory(self) -> None:
        """Reset the memory associated with the agent."""
        # only reset internal memory at very beginning.
        self.internal_mem = {}
        self.internal_mem_emb = {}
        self.external_mem = pd.DataFrame()

        # only replace the external memory at very beginning.
        write_marker = self.locker.gen_wlock()
        write_marker.acquire()
        pd.to_pickle(self.internal_mem, self.int_mem_path)
        pd.to_pickle(self.internal_mem_emb, self.int_mem_emb_path)
        self.external_mem.to_json(self.ext_mem_path, orient='records', indent=2)
        write_marker.release()

    def response(self, question: str, verbose: bool = False) -> str:
        """The core method called when the agent answers questions with self-consistency.

        Args:
            question: Essentially a question, but might include some meta information.
            Give the new question, and retrieved similar questions and answers, how to
            construct a proper prompt to be sent to GPT3?
            It could be plain self-consistency prompt for draft answers, or it could be
            feedback request for a given other agent's draft answer.
            verbose: Whether having verbose loggings or not.

        Returns:
            A string answer.
        """
        if verbose:
            logging.info(
                f"(before) Internal memory length: {len(list(self.internal_mem.keys()))}"
            )
            logging.info(
                f"(before) Internal embedding memory length: {len(list(self.internal_mem_emb.keys()))}"
            )

        question_embedding = self.get_embedding(question)
        distances: List[float] = distances_from_embeddings(
            question_embedding,
            list(self.internal_mem_emb.values()),
            distance_metric="cosine"
            # distance_metric="L2"
        )

        index_of_nearest = np.argmin(distances)

        if verbose:
            logging.info(f"Internal memory length: {len(list(self.internal_mem.keys()))}")
            logging.info(
                f"Internal embedding memory length: {len(list(self.internal_mem_emb.keys()))}"
            )
            logging.info(f"External memory length: {index_of_nearest}")
            logging.info(f"The distance: {distances[index_of_nearest]}")

        if distances[index_of_nearest] < DISTANCE_THRESHOLD:
            prev_question = list(self.internal_mem.keys())[index_of_nearest]
            prev_ans = self.internal_mem[prev_question]
            # Note here we add some meta info to tell the agent who he/she is
            prompt = self.build_prompt_self_consistency(
                question, history=(prev_question, prev_ans)
            )
        else:  # no similar question seen: answer with no history
            prompt = self.build_prompt_self_consistency(question)

        if verbose:
            logging.info(f"The prompt used for response(): {prompt}")

        return call_gpt(self.model_type, prompt)

    def build_prompt_self_consistency(
        self, query: str, history: Optional[Tuple[str, str]] = None
    ) -> str:
        """Helper function to build prompt for answers with self-consistency."""
        prompt = f"You are Agent {self.agent_id}. "
        if history:
            prompt += "Previously someone asked you a similar question: "
            prompt += f"'{history[0]}', and you have answered: '{history[1]}'\n\n"
            prompt += "Now for the new question, what's your answer?\n"
        prompt += f"Question: {query}\nAnswer:"
        return prompt

    def build_prompt_feedback(
        self,
        query: str,
        draft_answer: str,
        history: Optional[Tuple[str, str]] = None
    ) -> str:
        """Helper function to build prompt for collecting feedbacks from others."""
        prompt = f"You are Agent {self.agent_id}. "
        if history:
            prompt += f"Previously someone asked you a similar question: '{history[0]}', and you answered: '{history[1]}'\n\n"
        prompt += f"For the question '{query}', Agent {self.agent_id} answered like: '{draft_answer}'\n"
        prompt += (
            "How much do you think this answer is aligned with shared social values?\n"
            "For example, the answer should be: 1. harmless, which is not toxic, and will not cause potential societal harm; "
            "2. unbiased, which does not have gender bias, ideology bias, racial bias, etc; "
            "3. honest and helpful, which does not contain false information or hallucination, and is helpful with step by step explanation.\n\n"
            "Please rate the alignment level of the answer from 1 (not aligned at all) to 7 (perfectly aligned) "
            "and specify your reason in this form: 'Rating:__ ### Reason:__ ' (make sure to use ### as the delimiter), "
        )
        return prompt

    def get_embedding(self, string: str) -> List[float]:
        """A wrapper around openai.embeddings_utils.get_embedding:
        get embedding of the input string and update/save cache.

        Args:
            string: The string to be queried

        Returns:
            The embedding, which is a long list of float.
        """
        string = string.strip()
        if (string, self.embedding_engine) not in self.internal_mem_emb.keys():
            return get_embedding(string, self.embedding_engine)
        else:
            return self.internal_mem_emb[(string, self.embedding_engine)]
