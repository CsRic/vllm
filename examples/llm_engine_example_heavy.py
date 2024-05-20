import argparse
from typing import List, Tuple

from vllm import EngineArgs, LLMEngine, RequestOutput, SamplingParams

import random
import time
import numpy as np
import sys


def create_test_prompts() -> List[Tuple[str, SamplingParams]]:
    """Create a list of test prompts with their sampling parameters."""
    return [
        ("A robot may not injure a human being",
         SamplingParams(temperature=0.0, logprobs=1, prompt_logprobs=1)),
        ("To be or not to be,",
         SamplingParams(temperature=0.8, top_k=5, presence_penalty=0.2)),
        ("What is the meaning of life?",
         SamplingParams(n=2,
                        best_of=5,
                        temperature=0.8,
                        top_p=0.95,
                        frequency_penalty=0.1)),
        ("It is only with the heart that one can see rightly",
         SamplingParams(n=3, best_of=3, use_beam_search=True,
                        temperature=0.0)),
    ] * 1000

def create_test_prompt_nonsense(args: argparse.Namespace, user_id=0) -> List[Tuple[str, SamplingParams]]:
    length = args.input_len
    return (np.random.randint(10000, size=(length)).tolist(),
            SamplingParams(temperature=0.0, logprobs=1, prompt_logprobs=1, 
                        min_tokens=args.min_output_len, 
                        max_tokens=args.max_output_len),
            user_id,
        )


def process_requests(engine: LLMEngine,
                     test_prompts: List[Tuple[str, SamplingParams]]):
    """Continuously process a list of prompts and handle the outputs."""
    request_id = 0

    while test_prompts or engine.has_unfinished_requests():
        if test_prompts:
            prompt_token_ids, sampling_params, user_id = test_prompts.pop(0)
            engine.add_request(str(request_id), None, sampling_params, prompt_token_ids, user_id=user_id)
            request_id += 1

        request_outputs: List[RequestOutput] = engine.step()
        print(f"step at time {time.time()}, output num: {len(request_outputs)}")

        for request_output in request_outputs:
            if request_output.finished:
                print(f"finished: {request_output.request_id}")
                pass


def initialize_engine(args: argparse.Namespace) -> LLMEngine:
    """Initialize the LLMEngine from the command line arguments."""
    engine_args = EngineArgs.from_cli_args(args)
    return LLMEngine.from_engine_args(engine_args)


def main(args: argparse.Namespace):
    """Main function that sets up and runs the prompt processing."""
    engine = initialize_engine(args)
    test_prompts = []
    for i in range(args.test_num):
        for user_id in range(args.num_users):
            test_prompts.append(create_test_prompt_nonsense(args, user_id=user_id))
    process_requests(engine, test_prompts)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Demo on using the LLMEngine class directly')
    parser = EngineArgs.add_cli_args(parser)
    parser.add_argument('--num-users', 
                        type=int,
                        default=1)
    parser.add_argument('--min-output-len', type=int, default=64)
    parser.add_argument('--max-output-len', type=int, default=128)
    parser.add_argument('--input-len', type=int, default=128)
    parser.add_argument('--test-num', type=int, default=100)
    parser.add_argument('--snapshot', type=int, default=50)
    parser.add_argument('--interval', type=float, default=0)
    args = parser.parse_args()
    main(args)
