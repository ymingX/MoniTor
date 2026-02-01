#!/bin/bash
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1 --ntasks-per-node=1 --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --array=0-4%5
#SBATCH --output=output/01_caption_ucf_crime_%A_%a.out

# Set the UCF Crime directory
# ucf_crime_dir="/your/path/to/ucf_crime"
ucf_crime_dir="./dataset"

# Set paths
root_path="${ucf_crime_dir}/frames"
annotationfile_path="${ucf_crime_dir}/annotations/test.txt"
batch_size=16
frame_interval=15

# Qwen3-VL-4B local path is provided by environment variable MODEL_PATH
# Example (bash): export MODEL_PATH=/your/local/Qwen3-VL-4B
pretrained_model_names=(
    "$MODEL_PATH"
)

# Activate the virtual environment
# VENV_DIR="/path/to/venv/lavad"
# shellcheck source=/dev/null
#source "$VENV_DIR/bin/activate"
# name="your_file_name"
# Get the pretrained model name for the current task ID

pretrained_model_name="${pretrained_model_names[$SLURM_ARRAY_TASK_ID]}"
echo "Processing model: $pretrained_model_name"

# process output dir in .py instead of bash
# output_dir="${ucf_crime_dir}/captions/raw/${pretrained_model_name}/"

# Run the Python script with the specified parameters
python -m src.models.image_captioner \
    --root_path "$root_path" \
    --annotationfile_path "$annotationfile_path" \
    --batch_size "$batch_size" \
    --frame_interval "$frame_interval" \
    --pretrained_model_name "$pretrained_model_name" \
    # --output_dir "$output_dir"
