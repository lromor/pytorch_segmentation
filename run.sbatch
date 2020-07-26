#!/bin/bash
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=leonardo.romor@gmail.com
#SBATCH --time 8:00:00
#SBATCH --job-name segm
#SBATCH --ntasks=1
#SBATCH --partition=gpu_titanrtx
#SBATCH --gres=gpu:4

module purge
module load 2019
module load cuDNN/7.6.5.32-CUDA-10.1.243
module load NCCL/2.5.6-CUDA-10.1.243

nvidia-smi --query-gpu=gpu_name,gpu_bus_id,vbios_version --format=csv

PYTHONPATH=../ python3.7 train.py --config config.json
