#!/usr/bin/env bash
set -euo pipefail

echo "[1/4] Train unet_B..."
python train.py --model_name="unet_B" --save_dir="./checkpoints/B" --epochs=200
echo "Waiting 30 minutes before next command..."
sleep 30

echo "[2/4] Evaluate unet_B..."
python evaluate.py --model_name="unet_B" --weights="./checkpoints/B/best_model.pth" --save_dir="./predictions/B"
echo "Waiting 30 minutes before next command..."
sleep 30

echo "[3/4] Train unet_C..."
python train_C.py --epochs=200
echo "Waiting 30 minutes before next command..."
sleep 30

echo "[4/4] Evaluate unet_C..."
python evaluate_C.py

echo "All commands finished successfully."
