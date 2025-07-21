token=
model=meta-llama/Llama-3.2-1B-Instruct
volume=$PWD/data # share a volume with the 2
docker run --gpus all --shm-size 1g \
     -e HF_TOKEN=$token \
     -e CUDA_VISIBLE_DEVICES=6 \
     -p 8000:80 -v $volume:/data \
      ghcr.io/huggingface/text-generation-inference:3.3.0 \
    --model-id $model \
    --max-batch-prefill-tokens 32768 \
    --max-best-of 10 \
    --max-input-tokens 17000 \
    --max-total-tokens 32768