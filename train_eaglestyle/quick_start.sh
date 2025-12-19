#!/bin/bash

# Quick start script for MTP head training
# This script demonstrates the complete training pipeline with example configurations

set -e

# Configuration
BASE_MODEL="Qwen/Qwen2-7B"
INPUT_TEXT_FILE="./input_texts.txt"  # 需要提供，每行一个文档
COLLECTED_DATA_DIR="./mtp_collected_data"
TRAINING_OUTPUT_DIR="./mtp_head_checkpoint"

BLOCK_LENGTH=4
BATCH_SIZE=32
NUM_EPOCHS=3
LEARNING_RATE=1e-4

echo "=========================================="
echo "MTP Head Training Pipeline Quick Start"
echo "=========================================="

# Step 1: Data Collection
echo ""
echo "Step 1: Collecting data from base model..."
echo "Command:"
echo "python data_collection.py \\"
echo "    --base_model_path ${BASE_MODEL} \\"
echo "    --input_data_path ${INPUT_TEXT_FILE} \\"
echo "    --output_dir ${COLLECTED_DATA_DIR} \\"
echo "    --max_samples 10000 \\"
echo "    --sample_size 512 \\"
echo "    --stride 256"
echo ""
echo "Note: This step requires the base model and input text file."
echo "      Input file format: one document per line"
echo ""

read -p "Do you want to run data collection? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    python data_collection.py \
        --base_model_path ${BASE_MODEL} \
        --input_data_path ${INPUT_TEXT_FILE} \
        --output_dir ${COLLECTED_DATA_DIR} \
        --max_samples 10000 \
        --sample_size 512 \
        --stride 256
fi

# Step 2: Test Pipeline
echo ""
echo "Step 2: Testing the training pipeline..."
if [ -d "${COLLECTED_DATA_DIR}/collected_data_final" ]; then
    python test_pipeline.py \
        --data_dir "${COLLECTED_DATA_DIR}/collected_data_final" \
        --block_length ${BLOCK_LENGTH}
else
    echo "Collected data directory not found. Skipping test."
fi

# Step 3: Train MTP Head
echo ""
echo "Step 3: Training MTP head..."
echo "Command:"
echo "python train_mtp_head.py \\"
echo "    --train_data_dir ${COLLECTED_DATA_DIR}/collected_data_final \\"
echo "    --output_dir ${TRAINING_OUTPUT_DIR} \\"
echo "    --block_length ${BLOCK_LENGTH} \\"
echo "    --per_device_train_batch_size ${BATCH_SIZE} \\"
echo "    --num_train_epochs ${NUM_EPOCHS} \\"
echo "    --learning_rate ${LEARNING_RATE} \\"
echo "    --warmup_steps 500 \\"
echo "    --logging_steps 10 \\"
echo "    --save_steps 100"
echo ""

read -p "Do you want to run training? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    if [ -d "${COLLECTED_DATA_DIR}/collected_data_final" ]; then
        python train_mtp_head.py \
            --train_data_dir "${COLLECTED_DATA_DIR}/collected_data_final" \
            --output_dir ${TRAINING_OUTPUT_DIR} \
            --block_length ${BLOCK_LENGTH} \
            --per_device_train_batch_size ${BATCH_SIZE} \
            --num_train_epochs ${NUM_EPOCHS} \
            --learning_rate ${LEARNING_RATE} \
            --warmup_steps 500 \
            --logging_steps 10 \
            --save_steps 100 \
            --scheduler_type linear \
            --time_epsilon 0.01
    else
        echo "Error: Collected data directory not found!"
        exit 1
    fi
fi

echo ""
echo "=========================================="
echo "Training pipeline completed!"
echo "=========================================="
echo ""
echo "Trained model saved to: ${TRAINING_OUTPUT_DIR}"
echo ""
