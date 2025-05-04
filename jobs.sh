#!/bin/bash
### Sets the job's name.
#SBATCH --job-name=isic_unet_job

### Sets the job's output file and path.
#SBATCH --output=myFirstJob.out.%j

### Sets the job's error output file and path.
#SBATCH --error=myFirstJob.err.%j

### Requested number of nodes
#SBATCH -N 1

### Requested partition (choose the right one)
#SBATCH -p kimq

### Request 1 GPU
#SBATCH --gres=gpu:1

### Limit on the total run time (hh:mm:ss)
#SBATCH --time=05:00:00

echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

# Initialize Conda
echo "Activating A2Segmentation conda environment"
source ~/miniconda3/bin/activate
conda activate A2Segmentation

# Run your PyTorch U-Net training script
echo "Running Segmentation main.py"
python3 /home/<username>/main.py

# Deactivate the conda environment
echo "Deactivating environment"
conda deactivate

echo "Done."
