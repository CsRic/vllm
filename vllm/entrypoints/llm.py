from typing import List, Optional, Union

import torch
from tqdm import tqdm
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

from vllm.engine.arg_utils import EngineArgs
from vllm.engine.llm_engine import LLMEngine
from vllm.logger import init_logger
from vllm.lora.request import LoRARequest
from vllm.outputs import EmbeddingRequestOutput, RequestOutput
from vllm.pooling_params import PoolingParams
from vllm.sampling_params import SamplingParams
from vllm.sequence import MultiModalData
from vllm.usage.usage_lib import UsageContext
from vllm.utils import Counter

from vllm.core.user_log import UserLog
import time

logger = init_logger(__name__)

class LLM:
    """An LLM for generating texts from given prompts and sampling parameters.

    This class includes a tokenizer, a language model (possibly distributed
    across multiple GPUs), and GPU memory space allocated for intermediate
    states (aka KV cache). Given a batch of prompts and sampling parameters,
    this class generates texts from the model, using an intelligent batching
    mechanism and efficient memory management.

    NOTE: This class is intended to be used for offline inference. For online
    serving, use the `AsyncLLMEngine` class instead.
    NOTE: For the comprehensive list of arguments, see `EngineArgs`.

    Args:
        model: The name or path of a HuggingFace Transformers model.
        tokenizer: The name or path of a HuggingFace Transformers tokenizer.
        tokenizer_mode: The tokenizer mode. "auto" will use the fast tokenizer
            if available, and "slow" will always use the slow tokenizer.
        skip_tokenizer_init: If true, skip initialization of tokenizer and
            detokenizer. Expect valid prompt_token_ids and None for prompt
            from the input.
        trust_remote_code: Trust remote code (e.g., from HuggingFace) when
            downloading the model and tokenizer.
        tensor_parallel_size: The number of GPUs to use for distributed
            execution with tensor parallelism.
        dtype: The data type for the model weights and activations. Currently,
            we support `float32`, `float16`, and `bfloat16`. If `auto`, we use
            the `torch_dtype` attribute specified in the model config file.
            However, if the `torch_dtype` in the config is `float32`, we will
            use `float16` instead.
        quantization: The method used to quantize the model weights. Currently,
            we support "awq", "gptq", "squeezellm", and "fp8" (experimental).
            If None, we first check the `quantization_config` attribute in the
            model config file. If that is None, we assume the model weights are
            not quantized and use `dtype` to determine the data type of
            the weights.
        revision: The specific model version to use. It can be a branch name,
            a tag name, or a commit id.
        tokenizer_revision: The specific tokenizer version to use. It can be a
            branch name, a tag name, or a commit id.
        seed: The seed to initialize the random number generator for sampling.
        gpu_memory_utilization: The ratio (between 0 and 1) of GPU memory to
            reserve for the model weights, activations, and KV cache. Higher
            values will increase the KV cache size and thus improve the model's
            throughput. However, if the value is too high, it may cause out-of-
            memory (OOM) errors.
        swap_space: The size (GiB) of CPU memory per GPU to use as swap space.
            This can be used for temporarily storing the states of the requests
            when their `best_of` sampling parameters are larger than 1. If all
            requests will have `best_of=1`, you can safely set this to 0.
            Otherwise, too small values may cause out-of-memory (OOM) errors.
        enforce_eager: Whether to enforce eager execution. If True, we will
            disable CUDA graph and always execute the model in eager mode.
            If False, we will use CUDA graph and eager execution in hybrid.
        max_context_len_to_capture: Maximum context len covered by CUDA graphs.
            When a sequence has context length larger than this, we fall back
            to eager mode (DEPRECATED. Use `max_seq_len_to_capture` instead).
        max_seq_len_to_capture: Maximum sequence len covered by CUDA graphs.
            When a sequence has context length larger than this, we fall back
            to eager mode.
        disable_custom_all_reduce: See ParallelConfig
    """

    def __init__(
        self,
        model: str,
        tokenizer: Optional[str] = None,
        tokenizer_mode: str = "auto",
        skip_tokenizer_init: bool = False,
        trust_remote_code: bool = False,
        tensor_parallel_size: int = 1,
        dtype: str = "auto",
        quantization: Optional[str] = None,
        revision: Optional[str] = None,
        tokenizer_revision: Optional[str] = None,
        seed: int = 0,
        gpu_memory_utilization: float = 0.9,
        swap_space: int = 4,
        enforce_eager: bool = False,
        max_context_len_to_capture: Optional[int] = None,
        max_seq_len_to_capture: int = 8192,
        disable_custom_all_reduce: bool = False,
        **kwargs,
    ) -> None:
        if "disable_log_stats" not in kwargs:
            kwargs["disable_log_stats"] = True
        engine_args = EngineArgs(
            model=model,
            tokenizer=tokenizer,
            tokenizer_mode=tokenizer_mode,
            skip_tokenizer_init=skip_tokenizer_init,
            trust_remote_code=trust_remote_code,
            tensor_parallel_size=tensor_parallel_size,
            dtype=dtype,
            quantization=quantization,
            revision=revision,
            tokenizer_revision=tokenizer_revision,
            seed=seed,
            gpu_memory_utilization=gpu_memory_utilization,
            swap_space=swap_space,
            enforce_eager=enforce_eager,
            max_context_len_to_capture=max_context_len_to_capture,
            max_seq_len_to_capture=max_seq_len_to_capture,
            disable_custom_all_reduce=disable_custom_all_reduce,
            **kwargs,
        )
        if "use_csric_log" in kwargs:
            self.use_csric_log = kwargs["use_csric_log"]
        else:
            self.use_csric_log = False

        if self.use_csric_log:
        # and others...
            self.user_log = UserLog()
            self.step_count = 1
            self.snapshot = kwargs["snapshot"]
            self.num_users = kwargs["num_users"]

        self.llm_engine = LLMEngine.from_engine_args(
            engine_args, usage_context=UsageContext.LLM_CLASS, csric_log=self.user_log)
        self.request_counter = Counter()
        


    def get_tokenizer(
            self) -> Union[PreTrainedTokenizer, PreTrainedTokenizerFast]:
        return self.llm_engine.tokenizer.tokenizer

    def set_tokenizer(
        self,
        tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
    ) -> None:
        self.llm_engine.tokenizer.tokenizer = tokenizer

    def generate(
        self,
        prompts: Optional[Union[str, List[str]]] = None,
        sampling_params: Optional[Union[SamplingParams,
                                        List[SamplingParams]]] = None,
        prompt_token_ids: Optional[List[List[int]]] = None,
        use_tqdm: bool = True,
        lora_request: Optional[LoRARequest] = None,
        multi_modal_data: Optional[MultiModalData] = None,
        
        user_id: int = 0
    ) -> List[RequestOutput]:
        """Generates the completions for the input prompts.

        NOTE: This class automatically batches the given prompts, considering
        the memory constraint. For the best performance, put all of your prompts
        into a single list and pass it to this method.

        Args:
            prompts: A list of prompts to generate completions for.
            sampling_params: The sampling parameters for text generation. If
                None, we use the default sampling parameters. 
                When it is a single value, it is applied to every prompt. 
                When it is a list, the list must have the same length as the 
                prompts and it is paired one by one with the prompt.
            prompt_token_ids: A list of token IDs for the prompts. If None, we
                use the tokenizer to convert the prompts to token IDs.
            use_tqdm: Whether to use tqdm to display the progress bar.
            lora_request: LoRA request to use for generation, if any.
            multi_modal_data: Multi modal data.

        Returns:
            A list of `RequestOutput` objects containing the
            generated completions in the same order as the input prompts.
        """
        if sampling_params is None:
            # Use default sampling params.
            sampling_params = SamplingParams()

        requests_data = self._validate_and_prepare_requests(
            prompts,
            sampling_params,
            prompt_token_ids,
            lora_request,
            multi_modal_data,
            user_id = user_id,
        )

        # Add requests to the engine and run the engine
        for request_data in requests_data:
            self._add_request(**request_data)

        return self._run_engine(use_tqdm)

    def encode(
        self,
        prompts: Optional[Union[str, List[str]]] = None,
        pooling_params: Optional[Union[PoolingParams,
                                       List[PoolingParams]]] = None,
        prompt_token_ids: Optional[List[List[int]]] = None,
        use_tqdm: bool = True,
        lora_request: Optional[LoRARequest] = None,
        multi_modal_data: Optional[MultiModalData] = None,
    ) -> List[EmbeddingRequestOutput]:
        """Generates the completions for the input prompts.

        NOTE: This class automatically batches the given prompts, considering
        the memory constraint. For the best performance, put all of your prompts
        into a single list and pass it to this method.

        Args:
            prompts: A list of prompts to generate completions for.
            pooling_params: The pooling parameters for pooling. If None, we
                use the default pooling parameters.
            prompt_token_ids: A list of token IDs for the prompts. If None, we
                use the tokenizer to convert the prompts to token IDs.
            use_tqdm: Whether to use tqdm to display the progress bar.
            lora_request: LoRA request to use for generation, if any.
            multi_modal_data: Multi modal data.

        Returns:
            A list of `EmbeddingRequestOutput` objects containing the
            generated embeddings in the same order as the input prompts.
        """
        if pooling_params is None:
            # Use default pooling params.
            pooling_params = PoolingParams()

        requests_data = self._validate_and_prepare_requests(
            prompts,
            pooling_params,
            prompt_token_ids,
            lora_request,
            multi_modal_data,
        )

        # Add requests to the engine and run the engine
        for request_data in requests_data:
            self._add_request(**request_data)

        return self._run_engine(use_tqdm)

    def _validate_and_prepare_requests(
        self,
        prompts: Optional[Union[str, List[str]]],
        params: Union[Union[SamplingParams, PoolingParams],
                      List[Union[SamplingParams,
                                 PoolingParams]]],  # Unified parameter
        prompt_token_ids: Optional[List[List[int]]] = None,
        lora_request: Optional[LoRARequest] = None,
        multi_modal_data: Optional[MultiModalData] = None,
        user_id: int = 0
    ) -> List[dict]:
        """Validates and prepares request data for adding to the engine.

        Ensures prompts and token IDs are consistent, and returns a list of
        dictionaries with request data for further processing.
        """
        if prompts is None and prompt_token_ids is None:
            raise ValueError("Either prompts or prompt_token_ids must be "
                             "provided.")
        if self.llm_engine.model_config.skip_tokenizer_init \
            and prompts is not None:
            raise ValueError("prompts must be None if skip_tokenizer_init "
                             "is True")
        if isinstance(prompts, str):
            # Convert a single prompt to a list.
            prompts = [prompts]
        if (prompts is not None and prompt_token_ids is not None
                and len(prompts) != len(prompt_token_ids)):
            raise ValueError("The lengths of prompts and prompt_token_ids "
                             "must be the same.")

        if prompts is not None:
            num_requests = len(prompts)
        else:
            assert prompt_token_ids is not None
            num_requests = len(prompt_token_ids)

        if isinstance(params, list) and len(params) != num_requests:
            raise ValueError("The lengths of prompts and params "
                             "must be the same.")
        if multi_modal_data:
            multi_modal_data.data = multi_modal_data.data.to(torch.float16)

        # Add requests to the engine.
        requests_data = []
        for i in range(num_requests):
            prompt = prompts[i] if prompts is not None else None
            token_ids = None if prompt_token_ids is None else prompt_token_ids[
                i]

            multi_modal_item = MultiModalData(
                type=multi_modal_data.type,
                data=multi_modal_data.data[i].unsqueeze(0),
            ) if multi_modal_data else None

            requests_data.append({
                "prompt":
                prompt,
                "params":
                params[i] if isinstance(params, list) else params,
                "prompt_token_ids":
                token_ids,
                "lora_request":
                lora_request,
                "multi_modal_data":
                multi_modal_item,
                "user_id":
                user_id,
            })

        return requests_data

    def _add_request(
        self,
        prompt: Optional[str],
        params: Union[SamplingParams, PoolingParams],
        prompt_token_ids: Optional[List[int]],
        lora_request: Optional[LoRARequest] = None,
        multi_modal_data: Optional[MultiModalData] = None,
        user_id: int = 0
    ) -> None:
        request_id = str(next(self.request_counter))
        self.llm_engine.add_request(request_id,
                                    prompt,
                                    params,
                                    prompt_token_ids,
                                    lora_request=lora_request,
                                    multi_modal_data=multi_modal_data,
                                    user_id=user_id,)
        if self.use_csric_log:
            self.user_log.submit_request(int(request_id), user_id, len(prompt_token_ids), time.time())
    def _run_engine(
            self, use_tqdm: bool
    ) -> List[Union[RequestOutput, EmbeddingRequestOutput]]:
        # Initialize tqdm.
        if use_tqdm:
            num_requests = self.llm_engine.get_num_unfinished_requests()
            pbar = tqdm(
                total=num_requests,
                desc="Processed prompts",
                dynamic_ncols=True,
                postfix=f"Generation Speed: {0:.2f} toks/s",
            )
        # Run the engine.
        outputs: List[Union[RequestOutput, EmbeddingRequestOutput]] = []
        total_toks = 0
        while self.llm_engine.has_unfinished_requests():
            step_outputs = self.llm_engine.step()
            for output in step_outputs:
                is_finish = False
                if output.finished:
                    if self.use_csric_log:
                        self.step_count += 1
                        is_finish = True
                    outputs.append(output)
                    if use_tqdm:
                        if isinstance(output, RequestOutput):
                            # Calculate tokens only for RequestOutput
                            total_toks += sum(
                                len(stp.token_ids) for stp in output.outputs)
                            spd = total_toks / pbar.format_dict["elapsed"]
                            pbar.postfix = f"Generation Speed: {spd:.2f} toks/s"
                        pbar.update(1)
                if self.use_csric_log:
                    self.user_log.add_timestamp(int(output.request_id), time.time(), is_finish)
                    if self.step_count >= self.snapshot * self.num_users:
                        self.user_log.print_summary()
                        self.step_count = 1
                        
        if use_tqdm:
            pbar.close()
        # Sort the outputs by request ID.
        # This is necessary because some requests may be finished earlier than
        # its previous requests.
        outputs = sorted(outputs, key=lambda x: int(x.request_id))
        return outputs
