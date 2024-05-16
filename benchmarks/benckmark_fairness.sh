python fairness_example.py --model 'facebook/opt-1.3b' \
 -tp 4 \
 -fair \
 --enable-chunked-prefill \
 --gpu-memory-utilization 0.7 \
 --num-users 1 \
 --test-num 200 \
 --snapshot 50 \
 --min-output-len 64 \
 --max-output-len 64 \
 --input-len 64 \
 --interval 0.005



python fairness_example.py --model 'facebook/opt-1.3b' \
 -tp 4 \
 -fair \
 --gpu-memory-utilization 0.7 \
 --num-users 1 \
 --test-num 200 \
 --snapshot 50 \
 --min-output-len 64 \
 --max-output-len 64 \
 --input-len 64 \
 --interval 0.005



python fairness_example.py --model 'meta-llama/Llama-2-7b-hf' \
 -tp 4 \
 -fair \
 --enable-chunked-prefill \
 --gpu-memory-utilization 0.7 \
 --num-users 2 \
 --test-num 100 \
 --snapshot 25 \
 --min-output-len 128 \
 --max-output-len 128 \
 --input-len 128 \
 --interval 0.02


python fairness_example.py --model 'meta-llama/Llama-2-7b-hf' \
 -tp 4 \
 -fair \
 --gpu-memory-utilization 0.7 \
 --num-users 1 \
 --test-num 100 \
 --snapshot 25 \
 --min-output-len 128 \
 --max-output-len 128 \
 --input-len 128 \
 --interval 0.01
