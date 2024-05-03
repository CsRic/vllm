import argparse
from typing import List, Tuple

from vllm import EngineArgs, LLMEngine, RequestOutput, SamplingParams

import nltk
nltk.download('words')
from nltk.corpus import words
import random

def create_test_prompt_nonsense() -> List[Tuple[str, SamplingParams]]:
    english_words = words.words()
    length = random.randint(5, 100)
    user_id = random.randint(0,10)
    return (' '.join(random.choice(english_words) for _ in range(length)),
         SamplingParams(temperature=0.0, logprobs=1, prompt_logprobs=1),
         user_id)

def process_requests(engine: LLMEngine):
    """Continuously process a list of prompts and handle the outputs."""
    request_id = 0
    test_num = 100
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
    main(args)
