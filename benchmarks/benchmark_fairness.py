"""Benchmark the latency of processing a single batch of requests."""
import argparse
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from tqdm import tqdm

from vllm import LLM, SamplingParams
from vllm.model_executor.layers.quantization import QUANTIZATION_METHODS

import threading

def main(args: argparse.Namespace):
    print(args)
    llm = LLM(model=args.model,
              tokenizer=args.tokenizer,
              quantization=args.quantization,
              tensor_parallel_size=args.tensor_parallel_size,
              trust_remote_code=args.trust_remote_code,
              dtype=args.dtype,
              enforce_eager=args.enforce_eager,
              kv_cache_dtype=args.kv_cache_dtype,
              quantization_param_path=args.quantization_param_path,
              device=args.device,
              ray_workers_use_nsight=args.ray_workers_use_nsight,
              enable_chunked_prefill=args.enable_chunked_prefill,
              download_dir=args.download_dir,
              block_size=args.block_size,

              use_fairness_policy=args.use_fairness_policy,

              use_csric_log=True,
              snapshot=args.snapshot,
              num_users=args.num_users,
              )

    sampling_params = SamplingParams(
        n=args.n,
        temperature=0.0 if args.use_beam_search else 1.0,
        top_p=1.0,
        use_beam_search=args.use_beam_search,
        ignore_eos=True,
        max_tokens=args.max_output_len,
        min_tokens=args.min_output_len,
    )
    print(sampling_params)
    dummy_prompt_token_ids = np.random.randint(10000,
                                               size=(args.test_num,
                                                     args.input_len))
    test_prompts = dummy_prompt_token_ids.tolist()

    exit_event = threading.Event()

    def run_thread():
        while not exit_event.is_set():
            llm._run_engine(False)
        llm._run_engine(False)

    def request_thread():
        current_interval = 0.0
        last_request_time = time.time()
        while len(test_prompts) > 0:
            current_interval += time.time() - last_request_time
            last_request_time = time.time()
            while len(test_prompts) > 0 and current_interval >= args.interval:
                prompt_token_ids = test_prompts.pop(0)
                for i in range(args.num_users):
                    # llm.generate(prompt_token_ids=[prompt_token_ids],
                    #              sampling_params=sampling_params,
                    #              use_tqdm=False,
                    #              user_id=i,)
                    requests_data = llm._validate_and_prepare_requests(
                                                        prompts=None,
                                                        params = sampling_params,
                                                       prompt_token_ids=[prompt_token_ids],
                                                       user_id=i)
                    for request_data in requests_data:
                        llm._add_request(**request_data)
                current_interval -= args.interval
        exit_event.set()
    
    t1 = threading.Thread(target=run_thread)
    t2 = threading.Thread(target=request_thread)

    t1.start()
    t2.start()
    t2.join()
    t1.join()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Benchmark the latency of processing a single batch of '
        'requests till completion.')
    parser.add_argument('--model', type=str, default='facebook/opt-125m')
    parser.add_argument('--tokenizer', type=str, default=None)
    parser.add_argument('--quantization',
                        '-q',
                        choices=[*QUANTIZATION_METHODS, None],
                        default=None)
    parser.add_argument('--tensor-parallel-size', '-tp', type=int, default=1)
    parser.add_argument('--input-len', type=int, default=32)
    parser.add_argument('--output-len', type=int, default=128)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--n',
                        type=int,
                        default=1,
                        help='Number of generated sequences per prompt.')
    parser.add_argument('--use-beam-search', action='store_true')
    parser.add_argument('--num-iters-warmup',
                        type=int,
                        default=10,
                        help='Number of iterations to run for warmup.')
    parser.add_argument('--num-iters',
                        type=int,
                        default=30,
                        help='Number of iterations to run.')
    parser.add_argument('--trust-remote-code',
                        action='store_true',
                        help='trust remote code from huggingface')
    parser.add_argument(
        '--dtype',
        type=str,
        default='auto',
        choices=['auto', 'half', 'float16', 'bfloat16', 'float', 'float32'],
        help='data type for model weights and activations. '
        'The "auto" option will use FP16 precision '
        'for FP32 and FP16 models, and BF16 precision '
        'for BF16 models.')
    parser.add_argument('--enforce-eager',
                        action='store_true',
                        help='enforce eager mode and disable CUDA graph')
    parser.add_argument(
        "--kv-cache-dtype",
        type=str,
        choices=['auto', 'fp8'],
        default='auto',
        help=
        'Data type for kv cache storage. If "auto", will use model data type. '
        'FP8_E5M2 (without scaling) is only supported on cuda version greater '
        'than 11.8. On ROCm (AMD GPU), FP8_E4M3 is instead supported for '
        'common inference criteria.')
    parser.add_argument(
        '--quantization-param-path',
        type=str,
        default=None,
        help='Path to the JSON file containing the KV cache scaling factors. '
        'This should generally be supplied, when KV cache dtype is FP8. '
        'Otherwise, KV cache scaling factors default to 1.0, which may cause '
        'accuracy issues. FP8_E5M2 (without scaling) is only supported on '
        'cuda version greater than 11.8. On ROCm (AMD GPU), FP8_E4M3 is '
        'instead supported for common inference criteria.')
    parser.add_argument(
        '--profile',
        action='store_true',
        help='profile the generation process of a single batch')
    parser.add_argument(
        '--profile-result-dir',
        type=str,
        default=None,
        help=('path to save the pytorch profiler output. Can be visualized '
              'with ui.perfetto.dev or Tensorboard.'))
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help='device type for vLLM execution, supporting CUDA and CPU.')
    parser.add_argument('--block-size',
                        type=int,
                        default=16,
                        help='block size of key/value cache')
    parser.add_argument(
        '--enable-chunked-prefill',
        action='store_true',
        help='If True, the prefill requests can be chunked based on the '
        'max_num_batched_tokens')
    parser.add_argument(
        "--ray-workers-use-nsight",
        action='store_true',
        help="If specified, use nsight to profile ray workers",
    )
    parser.add_argument('--download-dir',
                        type=str,
                        default=None,
                        help='directory to download and load the weights, '
                        'default to the default cache dir of huggingface')
    parser.add_argument(
            '--gpu-memory-utilization',
            type=float,
            default=0.9,
            help='The fraction of GPU memory to be used for the model '
            'executor, which can range from 0 to 1. For example, a value of '
            '0.5 would imply 50%% GPU memory utilization. If unspecified, '
            'will use the default value of 0.9.')

    parser.add_argument('--use-fairness-policy', '-fair',
                        action='store_true')
    parser.add_argument('--num-users', 
                        type=int,
                        default=1)
    parser.add_argument('--min-output-len', type=int, default=1)
    parser.add_argument('--max-output-len', type=int, default=128)
    parser.add_argument('--test-num', type=int, default=100)
    parser.add_argument('--snapshot', type=int, default=50)
    parser.add_argument('--interval', type=float, default=0)

    args = parser.parse_args()
    main(args)
