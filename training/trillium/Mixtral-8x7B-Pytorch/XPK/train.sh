#!/bin/bash

# source /app/env.sh
# source env.sh
export ZONE=us-east5-c
export PROJECT=tpu-prod-env-automated
export TPU_TYPE=v6e-256
export NUM_SLICE=1
export CLUSTER_NAME=bodaborg-v6e-256 # use existing CLUSTER if you have

# Environment variables associated with training config.
export BATCH_PER_DEVICE=4
export SEQUENCE_LENGTH=4096
export MAX_STEP=50
export WORKLOAD_NAME=${USER}-xpk-${TPU_TYPE}-nov261457pm # Your workload name. Need to update for different run.
export BASE_DOCKER_IMAGE=us-central1-docker.pkg.dev/deeplearning-images/reproducibility/pytorch-tpu-mixtral:v1
export PROFILE_LOG_DIR=gs://manfei_bucket_automated # GCS bucket to store profile in form of gs://...
export HF_TOKEN=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 # Add your own Hugging face token to download model



# Extract the number after '-' in TPU_TYPE
TPU_NUM=$(echo "$TPU_TYPE" | grep -oP '(?<=-)\d+')
# echo $TPU_NUM

# Calculate GLOBAL_BATCH_SIZE
GLOBAL_BATCH_SIZE=$(( TPU_NUM * BATCH_PER_DEVICE * NUM_SLICE ))

export GLOBAL_BATCH_SIZE

echo "GLOBAL_BATCH_SIZE=$GLOBAL_BATCH_SIZE"

export PJRT_DEVICE=TPU
export XLA_USE_SPMD=1
export ENABLE_PJRT_COMPATIBILITY=true
export XLA_IR_DEBUG=1
export XLA_HLO_DEBUG=1
export PROFILE_EPOCH=0
export PROFILE_STEP=3
export PROFILE_DURATION_MS=100000
export PROFILE_LOGDIR=${PROFILE_LOG_DIR}
export XLA_PERSISTENT_CACHE_PATH=/app/xla_cache/
export TPU_LIBRARY_PATH=/workspace/_libtpu.so
export NUM_TPU_SLICE=1 # ${NUM_SLICE}
echo "NUM_TPU_SLICE=$NUM_TPU_SLICE"
export LIBTPU_INIT_ARGS="--xla_tpu_enable_flash_attention=false --xla_tpu_enable_async_collective_fusion=true --xla_tpu_enable_async_collective_fusion_fuse_all_gather=true --xla_tpu_enable_async_collective_fusion_multiple_steps=true --xla_tpu_overlap_compute_collective_tc=true --xla_enable_async_all_gather=true --xla_tpu_scoped_vmem_limit_kib=81920"

huggingface-cli login --token=${HF_TOKEN}

echo "CURRENT_PATH=$PWD"
ls -l
echo "all files"

# Note --per_device_train_batch_size is the global batch size since we overwrite the dataloader in the HF trainer.
python3 /workspace/transformers/examples/pytorch/language-modeling/run_clm.py \
--dataset_name=wikitext --dataset_config_name=wikitext-103-raw-v1 \
--per_device_train_batch_size=${GLOBAL_BATCH_SIZE} --do_train --output_dir=test-clm \
--overwrite_output_dir --config_name=./config.json \
--cache_dir=cache --tokenizer_name=mistralai/Mixtral-8x7B-v0.1 \
--block_size=${SEQUENCE_LENGTH} --optim=adafactor --save_strategy=no \
--logging_strategy=no --fsdp="full_shard" \
--fsdp_config=./fsdp_config.json --torch_dtype=bfloat16 \
--dataloader_drop_last=yes --max_steps=${MAX_STEP} --gmm --flash_attention
