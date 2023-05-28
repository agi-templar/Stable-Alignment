"""The script to collect data for social simulations.

Example:
    python collect_data.py --model_name 'gpt4' --world_ids "1, 2, 3, 4, 5"
"""

import glob
import math
from typing import Any, Dict, Sequence

import pandas as pd
from absl import app, flags

FLAGS = flags.FLAGS
CACHE_DIR_PREFIX: str = "./data/cache"

flags.DEFINE_string(
    'model_name',
    default=None,
    help='The model type of the social agents.',
)

flags.DEFINE_list(
    'world_ids',
    default=None,
    help='The IDs of the world that simulates human society. '
    'Takes a comma-separated list of strings',
)

flags.DEFINE_float(
    'self_critic_ratio',
    default=0.1,
    help='The ratio of self-critic data against the imitation data. '
    'e.g., 0.2 means sampling 20 self-critic samples if there are 100 imitation samples.',
)

flags.DEFINE_float(
    'realignment_ratio',
    default=0.1,
    help='The ratio of realignment data against the imitation data. '
    'e.g., 0.1 means sampling 10 self-critic samples if there are 100 imitation samples.',
)

flags.mark_flag_as_required('model_name')
flags.mark_flag_as_required('world_ids')

CRITIC_PROMPT = ' Please comment on the response in the input in terms of how well it is aligned with shared social values.'
REVISION_PROMPT = ' Please first common on this response, and then revise it to make it better aligned with shared social values.'
REVISION_START = ' Here is the revised response: '

PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
    "prompt_input_short":
    "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:",
    "prompt_no_input_short":
    "### Instruction:\n{instruction}\n\n### Response:",
}

prpt_inp, prpt_no_inpm, prpt_inp_s, prpt_no_inpm_s = PROMPT_DICT[
    "prompt_input"], PROMPT_DICT["prompt_no_input"], PROMPT_DICT[
        "prompt_input_short"], PROMPT_DICT["prompt_no_input_short"],


def fill_alpaca_template(df: pd.DataFrame) -> pd.DataFrame:
    """Fill the alignment data with Alpaca style template."""
    new_data = []
    for _, row in df.iterrows():
        if row["input"] == '':
            query = prpt_no_inpm_s.format_map({'instruction': row['instruction']})
        else:
            query = prpt_inp_s.format_map(
                {
                    'instruction': row['instruction'],
                    'input': row['input'],
                }
            )
        new_data.append(
            {
                'query': query,
                'response': row['output'],
                'score': row['rating'],
                'model': row['model']
            }
        )
    return pd.DataFrame(new_data)


def group_df_by_query(df: pd.DataFrame) -> pd.DataFrame:
    """Group df based on instruction and input."""
    df = df.drop_duplicates(subset=['response'])
    grouped_df = (
        df.groupby(['query']).agg(
            {
                'response': lambda x: x.tolist(),
                'score': lambda x: x.tolist(),
                'model': lambda x: x.tolist(),
            }
        ).rename(
            {
                'response': 'responses',
                'score': 'scores',
                'model': 'models',
            }, axis=1
        ).reset_index()
    )
    return grouped_df


def construct_data(df: pd.DataFrame, model_name: str) -> Dict[str, Sequence]:
    self_critic = []
    inst_following = []
    revision = []
    for _, row in df.iterrows():
        if not math.isnan(float(row['gen_moral_score_before'])):
            inst_following.append(
                {
                    'instruction': row['question'],
                    'input': '',
                    'output': row['draft_answer'],
                    'rating': int(row['gen_moral_score_before']),
                    'model': model_name,
                }
            )

        if not math.isnan(float(row['gen_moral_score_after'])):
            inst_following.append(
                {
                    'instruction': row['question'],
                    'input': '',
                    'output': row['revised_answer'],
                    'rating': int(row['gen_moral_score_after']),
                    'model': model_name,
                }
            )

    df['feedback'] = df.apply(
        lambda x: x['feedback'][:len(x['rating'])]
        if len(x['feedback']) > len(x['rating']) else x['feedback'],
        axis=1
    )

    feedback_df = df.explode(['feedback', 'rating'])

    for _, row in feedback_df.iterrows():
        if row['feedback'] and row['rating'] and not math.isnan(float(row['rating'])):
            self_critic.append(
                {
                    'instruction': row['question'] + CRITIC_PROMPT,
                    'input': row['draft_answer'],
                    'output': row['feedback'],
                    'rating': int(row['rating']),
                    'model': model_name,
                }
            )

        if row['gen_moral_score_after'] > row['gen_moral_score_before'] and \
                row['gen_moral_score_after'] > 4 and \
                not math.isnan(float(row['gen_moral_score_after'])):

            revision.append(
                {
                    'instruction':
                    row['question'] + ' ' + row['draft_answer'] + REVISION_PROMPT,
                    'input': '',
                    'output':
                    str(row['feedback']) + REVISION_START + row['revised_answer'],
                    'rating': int(row['gen_moral_score_after']),
                    'model': model_name,
                }
            )

    return {
        'self_critic': self_critic,
        'inst_following': inst_following,
        'revision': revision,
    }


def main(argv: Any) -> None:
    agents_self, agents_inst, agents_rev = [], [], []

    for world_id in FLAGS.world_ids:
        agent_paths = glob.glob(
            CACHE_DIR_PREFIX + f"/world_{world_id}/external_memory/*.jsonl"
        )

        for agent_path in agent_paths:
            df = pd.read_json(agent_path, orient='records')
            data_bundle = construct_data(df, FLAGS.model_name)

            agents_self.append(pd.DataFrame(data_bundle['self_critic']))
            agents_inst.append(pd.DataFrame(data_bundle['inst_following']))
            agents_rev.append(pd.DataFrame(data_bundle['revision']))

    all_agents_imit_df = group_df_by_query(fill_alpaca_template(pd.concat(agents_inst)))
    all_agents_self_df = group_df_by_query(fill_alpaca_template(pd.concat(agents_self)))
    all_agents_rev_df = group_df_by_query(fill_alpaca_template(pd.concat(agents_rev)))

    n_inst_data = len(all_agents_imit_df)
    n_self_data, n_rev_data = len(all_agents_self_df), len(all_agents_rev_df)
    n_self_data = min(int(n_inst_data * FLAGS.self_critic_ratio), n_self_data)
    n_rev_data = min(int(n_rev_data * FLAGS.realignment_ratio), n_rev_data)

    all_agents_self_df = all_agents_self_df.sample(n=n_self_data)
    all_agents_rev_df = all_agents_rev_df.sample(n=n_rev_data)

    fin_df = pd.concat([all_agents_imit_df, all_agents_self_df, all_agents_rev_df])

    fin_df.to_json(f"./data/{FLAGS.model_name}.json", orient='records', indent=2)


if __name__ == '__main__':
    app.run(main)
