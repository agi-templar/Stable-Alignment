"""Run inference on a trained model.

Make sure you have downloaded the model in the `model_path` directory.

Example:
    python stable_alignment/run_inference.py --model_path './models/socially-good-lm' --device 'cuda:0'
"""

import json
import os
from typing import Any, Dict, List, Optional

import torch
import transformers
from absl import app, flags
from colorama import Fore, Style

FLAGS = flags.FLAGS

transformers.logging.set_verbosity_error()

DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "</s>"
DEFAULT_UNK_TOKEN = "</s>"

flags.DEFINE_string(
    'model_path',
    default=None,
    help='The path to the trained model.',
)

flags.DEFINE_string(
    'device',
    default=None,
    help='The target GPU device. e.g., cuda:0',
)


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


def generate_prompt(instruction: str, input: Optional[str] = None) -> str:
    if input:
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"""
    else:
        return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Response:"""


def generate_with_prompt_batch(
    model: transformers.PreTrainedModel,
    device: str,
    tokenizer: transformers.PreTrainedTokenizer,
    instructs: List[str],
    inputs: Optional[List[str]] = None,
    batch_size: int = 32,
    use_prompt: bool = True,
    output_path: Optional[str] = None
) -> List[str]:
    if inputs is None or len(inputs) == 0:
        print("inputs is None. Skip it.")
        inputs = [None] * len(instructs)

    results = []

    if output_path and os.path.exists(output_path):
        with open(output_path, 'r') as f:
            lines = f.readlines()
        lines = [line for line in lines if line]
        cnt = len(lines)
        print(f'Skip first {cnt} lines.')
        instructs = instructs[cnt:]
        inputs = inputs[cnt:]

    for batch_start in range(0, len(instructs), batch_size):
        batch_end = batch_start + batch_size
        batch_instructs = instructs[batch_start:batch_end]
        batch_inputs = inputs[batch_start:batch_end]

        batch_prompts = [
            generate_prompt(instr, inp) if use_prompt else instr
            for instr, inp in zip(batch_instructs, batch_inputs)
        ]

        print(Fore.GREEN + "Let's see one resulting prompt:" + Style.RESET_ALL)
        print(batch_prompts[0])

        encoded_inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True)
        input_ids = encoded_inputs["input_ids"].to(device)
        attention_mask = encoded_inputs["attention_mask"].to(device)

        if input_ids.shape[1] > 100:
            input_ids = input_ids[:, -100:]
            attention_mask = attention_mask[:, -100:]

        with torch.no_grad():
            generation_output = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=256,
                temperature=0.7,
                top_p=0.9,
                num_beams=1,
                do_sample=True,
                no_repeat_ngram_size=2,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
                return_dict_in_generate=True,
                output_scores=True
            )

        for seq in generation_output.sequences:
            output = tokenizer.decode(seq)
            if use_prompt:
                try:
                    res = output.split("### Response:")[1].strip()
                    res = res.split("### Instruction:")[0].strip()
                except BaseException:
                    res = ''
            else:
                res = output

            print(Fore.YELLOW + "Let's see one generation output:" + Style.RESET_ALL)
            print(res)

            results.append(res)
            if output_path:
                with open(output_path, 'a+') as f:
                    f.write(
                        json.dumps({
                            'response': res.split('</s>')[0],
                        }).strip() + "\n"
                    )

    results = [response.split('</s>')[0] for response in results]
    return results


def main(argv: Any) -> None:
    model = transformers.AutoModelForCausalLM.from_pretrained(FLAGS.model_path)
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        FLAGS.model_path,
        padding_side="left",  # for batch decode
        use_fast=False,
    )
    if tokenizer.pad_token is None:
        smart_tokenizer_and_embedding_resize(
            special_tokens_dict=dict(pad_token=DEFAULT_PAD_TOKEN),
            tokenizer=tokenizer,
            model=model,
        )
    tokenizer.add_special_tokens(
        {
            "eos_token": DEFAULT_EOS_TOKEN,
            "bos_token": DEFAULT_BOS_TOKEN,
            "unk_token": DEFAULT_UNK_TOKEN,
        }
    )

    model = model.to(FLAGS.device)
    model.eval()

    while True:
        inst = input('Please input your instruction:')
        inp = input('Please input your input (skip by pressing enter if no input):')
        res = generate_with_prompt_batch(
            model, FLAGS.device, tokenizer, [inst], [inp], batch_size=1, use_prompt=True
        )
        print(res[0])


if __name__ == '__main__':
    app.run(main)
