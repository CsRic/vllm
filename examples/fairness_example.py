import argparse
from typing import List, Tuple

from vllm import EngineArgs, LLMEngine, RequestOutput, SamplingParams
from vllm.model_executor.layers.quantization import QUANTIZATION_METHODS

from vllm.core.user_log import UserLog

import random
import time
import numpy as np
import sys

def create_test_prompt_nonsense(args: argparse.Namespace, user_id=0) -> List[Tuple[str, SamplingParams]]:
    length = args.input_len
    return (np.random.randint(10000, size=(length)).tolist(),
            SamplingParams(temperature=0.0, logprobs=1, prompt_logprobs=1, 
                        min_tokens=args.min_output_len, 
                        max_tokens=args.max_output_len),
            user_id,
        )

def process_requests(engine: LLMEngine, args: argparse.Namespace):
    """Continuously process a list of prompts and handle the outputs."""
    request_id = 0
    test_prompts = []
    for i in range(args.test_num):
        for user_id in range(args.num_users):
            test_prompts.append(create_test_prompt_nonsense(args, user_id))
    user_log = UserLog()
    current_interval = 0.0
    last_request_time = time.time()
    step_count = 1 # when accumulated to args.snapshot * args.num_users, reset to 1

    while len(test_prompts) > 0 or engine.has_unfinished_requests():
        current_interval += time.time() - last_request_time
        last_request_time = time.time()
        while len(test_prompts) > 0 and current_interval >= args.interval:
            for i in range(args.num_users):
                # each user send one request per interval
                prompt_token_ids, sampling_params, user_id = test_prompts.pop(0)
                user_log.submit_request(request_id, user_id, len(prompt_token_ids), time.time())
                engine.add_request(str(request_id), None, sampling_params, prompt_token_ids, user_id=user_id)
                request_id += 1
            current_interval -= args.interval
        if engine.has_unfinished_requests():
            request_outputs: List[RequestOutput] = engine.step()
            for request_output in request_outputs:
                # one token generated
                is_finish = False 
                if request_output.finished:
                    step_count += 1
                    is_finish = True
                    sys.stdout.write(f"\rfinish {request_output.request_id}")
                    sys.stdout.flush()
                user_log.add_timestamp(int(request_output.request_id), time.time(), is_finish)
                if(step_count >= args.snapshot * args.num_users):
                    user_log.print_summary()
                    step_count = 1
                    # user_log.clean_finished()

    user_log.print_summary()

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

    # 'facebook/opt-1.3b'
    # 'meta-llama/Llama-2-7b-hf'

    main(args)
