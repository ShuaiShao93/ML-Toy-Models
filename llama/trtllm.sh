# On a node with 2 GPUs

sudo apt install git-lfs
git lfs install

pip3 install tensorrt_llm==0.15.0 --extra-index-url https://pypi.nvidia.com
git clone https://huggingface.co/unsloth/Meta-Llama-3.1-8B-Instruct
git clone -b v0.15.0 https://github.com/NVIDIA/TensorRT-LLM.git

git clone https://huggingface.co/xxx/llama-3.1-8b-finetuned-lora-weights 

# bf16 with tp, lora
python3 TensorRT-LLM/examples/llama/convert_checkpoint.py --model_dir ./Meta-Llama-3.1-8B-Instruct --output_dir ./tllm_8b_checkpoint_1gpu_bf16 --dtype bfloat16 --tp_size=2
trtllm-build --checkpoint_dir ./tllm_8b_checkpoint_1gpu_bf16 --output_dir ./tmp/llama/8B/trt_engines/bf16/1-gpu  --gpt_attention_plugin auto  --gemm_plugin auto  --max_num_tokens 128000 --max_batch_size 8 --logits_dtype=float32 --gather_generation_logits --kv_cache_type=paged --lora_plugin auto --lora_dir llama-3.1-8b-finetuned-lora-weights

# int8 with tp, lora
python3 TensorRT-LLM/examples/llama/convert_checkpoint.py --model_dir ./Meta-Llama-3.1-8B-Instruct --output_dir ./tllm_8b_checkpoint_1gpu_int8  --dtype bfloat16 --tp_size=2  --use_weight_only   --weight_only_precision int8
trtllm-build --checkpoint_dir ./tllm_8b_checkpoint_1gpu_int8 --output_dir ./tmp/llama/8B/trt_engines/int8/1-gpu  --gpt_attention_plugin auto  --gemm_plugin auto  --max_num_tokens 128000 --max_batch_size 8 --logits_dtype=float32 --gather_generation_logits --kv_cache_type=paged --lora_plugin auto --lora_dir llama-3.1-8b-finetuned-lora-weights

# run
python3 TensorRT-LLM/examples/run.py --engine_dir=./tmp/llama/8B/trt_engines/bf16/1-gpu --max_output_len 1 --max_input_length=100000 --run_profiling --tokenizer_dir ./Meta-Llama-3.1-8B-Instruct --input_file 15k-tokens.txt --lora_dir llama-3.1-8b-finetuned-lora-weights --lora_task_uids 1

# benchmark
python TensorRT-LLM/benchmarks/python/benchmark.py -m dec --engine_dir ./tmp/llama/8B/trt_engines/bf16/1-gpu/ --dtype dfloat16  --batch_size "1;2;4" --input_output_len "600,1;1200,1"

#triton
git clone -b v0.15.0 https://github.com/triton-inference-server/tensorrtllm_backend.git
cp ./tmp/llama/8B/trt_engines/bf16/1-gpu/* tensorrtllm_backend/all_models/inflight_batcher_llm/tensorrt_llm/1/

HF_LLAMA_MODEL=./Meta-Llama-3.1-8B-Instruct
ENGINE_PATH=tensorrtllm_backend/all_models/inflight_batcher_llm/tensorrt_llm/1
BATCH_SIZE=8 

cd tensorrtllm_backend
git restore all_models/inflight_batcher_llm
python3 tools/fill_template.py -i all_models/inflight_batcher_llm/preprocessing/config.pbtxt tokenizer_dir:${HF_LLAMA_MODEL},triton_max_batch_size:${BATCH_SIZE},preprocessing_instance_count:1
python3 tools/fill_template.py -i all_models/inflight_batcher_llm/postprocessing/config.pbtxt tokenizer_dir:${HF_LLAMA_MODEL},triton_max_batch_size:${BATCH_SIZE},postprocessing_instance_count:1
python3 tools/fill_template.py -i all_models/inflight_batcher_llm/tensorrt_llm_bls/config.pbtxt triton_max_batch_size:${BATCH_SIZE},decoupled_mode:False,bls_instance_count:1,accumulate_tokens:False
python3 tools/fill_template.py -i all_models/inflight_batcher_llm/ensemble/config.pbtxt triton_max_batch_size:${BATCH_SIZE}
python3 tools/fill_template.py -i all_models/inflight_batcher_llm/tensorrt_llm/config.pbtxt triton_backend:tensorrtllm,triton_max_batch_size:${BATCH_SIZE},decoupled_mode:False,max_beam_width:1,engine_dir:${ENGINE_PATH},max_tokens_in_paged_kv_cache:16384,exclude_input_in_output:True,enable_kv_cache_reuse:False,batching_strategy:inflight_fused_batching,encoder_input_features_data_type:TYPE_FP16,max_queue_delay_microseconds:0,lora_cache_gpu_memory_fraction:0.1

cd ..
docker run -it --rm --gpus all --network host --shm-size=1g \
-v $(pwd):/workspace \
--workdir /workspace \
nvcr.io/nvidia/tritonserver:24.11-trtllm-python-py3

rm /workspace/logs.txt
python3 tensorrtllm_backend/scripts/launch_triton_server.py --model_repo tensorrtllm_backend/all_models/inflight_batcher_llm --world_size 2 --log --log-file /workspace/logs.txt

# curl to test triton
curl -X POST localhost:8000/v2/models/ensemble/generate -d \
'{
"text_input": "How do I count to nine in French?",
"parameters": {
"max_tokens": 100,
"bad_words":[""],
"stop_words":[""]
}
}' 

# script to test triton
python3 TensorRT-LLM/examples/hf_lora_convert.py -i llama-3.1-8b-finetuned-lora-weights -o llama-3.1-8b-finetuned-lora-weights-np --storage-type float16
python3 tensorrtllm_backend/inflight_batcher_llm/client/end_to_end_grpc_client.py -o 1 -p '["This is a test"]' --lora-path ./llama-3.1-8b-finetuned-lora-weights-np --lora-task-id 1

# benchmark triton
docker run --rm -it -v $(pwd):/workspace --net host nvcr.io/nvidia/tritonserver:24.11-py3-sdk 
perf_analyzer     -m ensemble     -u localhost:8000     --measurement-mode="count_windows"      --input-data=benchmark.json     --measurement-request-count="10"     -b "1"    --stability-percentage="10" --concurrency-range=4

# triton metrics
curl http://localhost:8002/metrics 
