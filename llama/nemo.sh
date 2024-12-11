# Install ngc cli first: https://org.ngc.nvidia.com/setup/installers/cli

# Start server
ngc registry model download-version nvidia/nemo/llama-3-8b-instruct-nemo:1.0
docker run --gpus all --network host -it --rm --shm-size=4g -p 8000:8000     -v ${PWD}/8b_instruct_nemo_bf16.nemo:/opt/checkpoints/8b_instruct_nemo_bf16.nemo     -w /opt/NeMo     nvcr.io/nvidia/nemo:24.09
 
python scripts/deploy/nlp/deploy_triton.py \
    --nemo_checkpoint /opt/checkpoints/8b_instruct_nemo_bf16.nemo \
    --model_type llama \
    --triton_model_name llama31_8b \
    --tensor_parallelism_size 1 \
    --max_input_len 1000000 \
    --max_batch_size 8 \
    --max_num_tokens 96000

# In another terminal, run query
docker run --gpus all --network host -it -v ~:/home -w /opt/NeMo nvcr.io/nvidia/nemo:24.09
python scripts/deploy/nlp/query.py -mn llama31_8b -pf /home/15k-tokens.txt --max_output_len 1
