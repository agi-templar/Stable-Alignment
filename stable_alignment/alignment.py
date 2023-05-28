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
"""Stable Alignment algorithm."""

import copy
import io
import json
import random
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple, Union

import pandas as pd
import torch
import transformers
from absl import logging
from torch.nn import CrossEntropyLoss
from torch.utils.data import Dataset
from transformers import Trainer

torch.autograd.set_detect_anomaly(True)

logging.set_verbosity("info")
logging.set_stderrthreshold("info")


def load_json_file(f: Any, mode: str = "r") -> Dict[str, Any]:
    """Load a .json file into a dictionary."""
    if not isinstance(f, io.IOBase):
        f = open(f, mode=mode)
    jdict = json.load(f)
    f.close()
    return jdict


IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "</s>"
DEFAULT_UNK_TOKEN = "</s>"


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")


@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})
    stop_response: bool = field(default=False)
    num_comp: int = field(default=3)


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={
            "help":
            "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    rating_scale: int = field(default=7)
    margin: float = field(default=1.0)
    max_flow: bool = field(default=False)
    ratio: float = field(default=0.5)


def safe_save_model_for_hf_trainer(
    trainer: transformers.Trainer, output_dir: str
) -> None:
    """Collects the state dict and dump to disk."""
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
) -> None:
    """Resize tokenizer and embedding.
    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True
        )
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True
        )

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def _tokenize_fn(
    strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer
) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ) for text in strings
    ]
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item()
        for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


class AlignmentDataset(Dataset):
    """Dataset for alignment training."""

    def __init__(self, data_path: str):
        super(AlignmentDataset, self).__init__()
        logging.info("Loading data...")
        self.data = pd.read_json(data_path, orient='records').to_dict('records')

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, i: int) -> dict:
        return dict(input_ids=self.data[i])


def _single_tokenize(
    text: str,
    tokenizer: transformers.PreTrainedTokenizer,
    max_len: int = 512
) -> torch.Tensor:
    if max_len is None:
        max_len = tokenizer.model_max_length
    toked = tokenizer(
        text,
        return_tensors="pt",
        padding="longest",
        max_length=max_len,
        truncation=True,
    )
    return toked['input_ids'][0]


def stop_response(res: str) -> str:
    stops = ['\n\nHuman:', '\n\nAssistant:', '\n\nhuman:', '\n\nassistant:']
    for stop in stops:
        if res.find(stop) >= 0:
            res = res[:res.find(stop)].strip()
    return res


@dataclass
class DataCollatorForAlignmentDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer
    stop_response: bool
    num_comp: int

    def __call__(self, instances: Iterable[Any]) -> Dict[str, torch.Tensor]:
        all_scores = []
        input_ids = []
        labels = []
        for _, instance in enumerate(instances):
            data_bundle = instance['input_ids']
            query = data_bundle['query']
            responses = data_bundle['responses']
            scores = data_bundle['scores']

            pairs = random.sample(
                list(zip(responses, scores)), min(len(responses), self.num_comp)
            )  # pick 3 random pairs
            responses, scores = zip(*pairs)  # separate the pairs

            all_scores.append([int(sc) for sc in scores])

            examples = [query + t for t in responses]
            source_len = _tokenize_fn([query], self.tokenizer)["input_ids_lens"][0]
            input_ids = _tokenize_fn(examples, self.tokenizer)["input_ids"]

            labels = copy.deepcopy(input_ids)
            for label in labels:
                label[:source_len] = IGNORE_INDEX

        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=IGNORE_INDEX
        )

        return dict(
            input_ids=input_ids,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
            labels=labels,
            scores=torch.FloatTensor(all_scores),
        )


def make_supervised_data_module(
    tokenizer: transformers.PreTrainedTokenizer,
    data_args: DataArguments,
) -> Dict:
    """Make dataset and collator for alignment training."""
    train_dataset = AlignmentDataset(data_path=data_args.data_path)
    data_collator = DataCollatorForAlignmentDataset(
        tokenizer=tokenizer,
        stop_response=data_args.stop_response,
        num_comp=data_args.num_comp
    )
    return dict(
        train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator
    )


class AlignmentTrainer(Trainer):

    def stable_alignment(
        self, logits: torch.Tensor, labels: torch.Tensor, feedback_scores: torch.Tensor
    ) -> torch.Tensor:
        # Calculate the SFT loss
        sorted_ratings, indices = torch.sort(feedback_scores.squeeze(), descending=True)
        best_idx = indices[0] if indices.dim() != 0 else indices.item()
        best_score = sorted_ratings[0] if indices.dim() != 0 else sorted_ratings.item()
        loss_fct = CrossEntropyLoss(ignore_index=IGNORE_INDEX)

        # Calculate the penalty from low-rating responses.
        batch_losses = []
        for logit, label in zip(logits, labels):
            batch_losses.append(loss_fct(logit.view(-1, logits.size(-1)), label.view(-1)))
        batch_loss = torch.stack(batch_losses, dim=0)

        # Modulate the penalty by the difference in ratings.
        min_loss = batch_loss[best_idx]
        neg_losses = []
        if indices.dim() != 0 and indices.size(-1) > 1:
            for idx in indices[1:]:
                margin = (
                    best_score - sorted_ratings[idx]
                ) / self.args.rating_scale * self.args.margin
                neg_loss = min_loss - batch_loss[idx] + margin
                neg_losses.append(neg_loss)

        if len(neg_losses) > 0:
            neg_losses_ts = torch.stack(neg_losses)
            if self.args.max_flow:
                diff = torch.max(torch.max(neg_losses_ts), torch.tensor(0.0).cuda())
            else:
                diff = torch.max(neg_losses_ts.mean(), torch.tensor(0.0).cuda())
        else:
            diff = torch.tensor(0.0).cuda()

        return min_loss + self.args.ratio * diff

    def compute_loss(
        self,
        model: transformers.PreTrainedModel,
        inputs: Dict[str, torch.Tensor],
        return_outputs: bool = False
    ) -> Union[float, Tuple[float, Dict[str, torch.Tensor]]]:
        input_ids = inputs.get('input_ids')
        attention_mask = inputs.get('attention_mask')
        labels = inputs.get('labels')
        feedback_scores = inputs.get('scores')  # 1 * (batch * cand)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True
        )  # (batch * cand) * L * V

        logits = outputs['logits']

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        loss = self.stable_alignment(shift_logits, shift_labels, feedback_scores)
        return (loss, {'outputs': outputs}) if return_outputs else loss


def train() -> None:
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    model = transformers.LlamaForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
    )

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )

    if tokenizer.pad_token is None:
        smart_tokenizer_and_embedding_resize(
            special_tokens_dict=dict(pad_token=DEFAULT_PAD_TOKEN),
            tokenizer=tokenizer,
            model=model,
        )
    # if "llama" in model_args.model_name_or_path:
    tokenizer.add_special_tokens(
        {
            "eos_token": DEFAULT_EOS_TOKEN,
            "bos_token": DEFAULT_BOS_TOKEN,
            "unk_token": DEFAULT_UNK_TOKEN,
        }
    )

    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)
    trainer = AlignmentTrainer(
        model=model, tokenizer=tokenizer, args=training_args, **data_module
    )
    trainer.train()
    trainer.save_state()
    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()
