import argparse
from typing import List, Tuple

from vllm import EngineArgs, LLMEngine, RequestOutput, SamplingParams
from vllm.model_executor.layers.quantization import QUANTIZATION_METHODS

import nltk
nltk.download('words')
from nltk.corpus import words
import random
import time
import numpy as np


test_num = 500


def create_test_prompt_nonsense(args: argparse.Namespace) -> List[Tuple[str, SamplingParams]]:
    english_words = words.words()
    length = args.input_len
    user_id = random.randint(0,args.num_users-1)
    return (np.random.randint(10000, size=(length)).tolist(),
            SamplingParams(temperature=0.0, logprobs=1, prompt_logprobs=1, 
                        min_tokens = args.min_output_len, 
                        max_tokens=args.max_output_len),
            user_id,
        )

def process_requests(engine: LLMEngine, args: argparse.Namespace):
    """Continuously process a list of prompts and handle the outputs."""
    request_id = 0
    test_prompts = []
    for i in range(test_num):
        test_prompts.append(create_test_prompt_nonsense(args))

    while test_prompts or engine.has_unfinished_requests():
        if test_prompts:
            prompt_token_ids, sampling_params, user_id = test_prompts.pop(0)
            engine.add_request(str(request_id), None, sampling_params, prompt_token_ids, user_id=user_id)
            request_id += 1

        request_outputs: List[RequestOutput] = engine.step()

        for request_output in request_outputs:
            if request_output.finished:
                # print(request_output)
                pass


def initialize_engine(args: argparse.Namespace) -> LLMEngine:
    """Initialize the LLMEngine from the command line arguments."""
    engine_args = EngineArgs.from_cli_args(args)
    engine_args.disable_log_stats = True
    return LLMEngine.from_engine_args(engine_args)


def main(args: argparse.Namespace):
    # nltk.download('words')

    """Main function that sets up and runs the prompt processing."""
    engine = initialize_engine(args)
    process_requests(engine, args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Demo on using the LLMEngine class directly')
    parser = EngineArgs.add_cli_args(parser)

    parser.add_argument('--use-fairness-policy', '-fair',
                        action='store_true')
    parser.add_argument('--num-users', 
                        type=int,
                        default=1)
    parser.add_argument('--min-output-len', type=int, default=1)
    parser.add_argument('--max-output-len', type=int, default=128)
    parser.add_argument('--input-len', type=int, default=128)
    args = parser.parse_args()

    # 'facebook/opt-1.3b'
    # 'meta-llama/Llama-2-7b-hf'

    main(args)
