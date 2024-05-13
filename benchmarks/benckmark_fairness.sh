python fairness_example.py --model 'meta-llama/Llama-2-7b-hf' -tp 4 \
    --enable-chunked-prefill \
    -fair \
    --gpu-memory-utilization 0.8 \
    --num-users 1 \
    --test-num 200 \
    --warm-up 50 \
    --min-output-len 64 \
    --max-output-len 128 \
    --input-len 128


python fairness_example.py --model 'meta-llama/Llama-2-7b-hf' -tp 4 \
    -fair \
    --gpu-memory-utilization 0.8 \
    --num-users 1 \
    --test-num 200 \
    --warm-up 50 \
    --min-output-len 64 \
    --max-output-len 128 \
    --input-len 128


python fairness_example.py --model 'meta-llama/Llama-2-7b-hf' -tp 4 \
    --enable-chunked-prefill \
    --gpu-memory-utilization 0.8 \
    --num-users 1 \
    --test-num 200 \
    --warm-up 50 \
    --min-output-len 64 \
    --max-output-len 128 \
    --input-len 128