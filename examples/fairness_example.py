import argparse
from typing import List, Tuple

from vllm import EngineArgs, LLMEngine, RequestOutput, SamplingParams

import nltk
nltk.download('words')
from nltk.corpus import words
import random
import time

num_users = 1
min_request_len = 90
max_request_len = 100
min_generation_len = 50
max_generation_len = 200
test_num = 1000


def create_test_prompt_nonsense() -> List[Tuple[str, SamplingParams]]:
    english_words = words.words()
    length = random.randint(min_generation_len, max_generation_len)
    user_id = random.randint(0,num_users-1)
    return (' '.join(random.choice(english_words) for _ in range(length)),
         SamplingParams(temperature=0.0, logprobs=1, prompt_logprobs=1, 
                        min_tokens = min_generation_len, 
                        max_tokens=max_generation_len),
         user_id)

def process_requests(engine: LLMEngine):
    """Continuously process a list of prompts and handle the outputs."""
    request_id = 0
    test_prompts = []
    for i in range(test_num):
        test_prompts.append(create_test_prompt_nonsense())

    while test_prompts or engine.has_unfinished_requests():
        if test_prompts:
            prompt, sampling_params, user_id = test_prompts.pop(0)
            engine.add_request(str(request_id), prompt, sampling_params, user_id=user_id)
            request_id += 1

        request_outputs: List[RequestOutput] = engine.step()

        for request_output in request_outputs:
            if request_output.finished:
                # print(request_output)
                pass


def initialize_engine(args: argparse.Namespace) -> LLMEngine:
    """Initialize the LLMEngine from the command line arguments."""
    engine_args = EngineArgs.from_cli_args(args)
    engine_args.enable_chunked_prefill = True
    engine_args.disable_log_stats = True
    engine_args.use_fairness_policy = False
    return LLMEngine.from_engine_args(engine_args)


def main(args: argparse.Namespace):
    nltk.download('words')

    """Main function that sets up and runs the prompt processing."""
    engine = initialize_engine(args)
    process_requests(engine)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Demo on using the LLMEngine class directly')
    parser = EngineArgs.add_cli_args(parser)
    args = parser.parse_args()

    # args.model = 'facebook/opt-1.3b'
    args.model = 'meta-llama/Llama-2-7b-hf'
    args.tensor_parallel_size = 1

    main(args)
