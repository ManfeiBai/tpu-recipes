#!/bin/bash

# Environment variables associated with XPK on GCP.
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
export HF_TOKEN=hf_ABC # Add your own Hugging face token to download model
