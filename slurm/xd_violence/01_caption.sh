#!/bin/bash
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1 --ntasks-per-node=1 --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --array=0-4%5
#SBATCH --output=output/01_caption_xd_violence_%A_%a.out

# Set the XD-Violence directory
xd_violence_dir="/mnt/d/qsh/1_data/xd_violence"

# Set paths
root_path="${xd_violence_dir}/frames"
annotationfile_path="${xd_violence_dir}/annotations/anomaly_test.txt"
batch_size=128
frame_interval=16

# Qwen3-VL-4B local path is provided by environment variable MODEL_PATH
# Example (bash): export MODEL_PATH=/your/local/Qwen3-VL-4B
pretrained_model_names=(
    "$MODEL_PATH"
)

# mingX: No need for this step. MoniTor's requirement.txt already includes all necessary packages.
# Activate the virtual environment
VENV_DIR="/path/to/venv/lavad"
# shellcheck source=/dev/null
#source "$VENV_DIR/bin/activate"

# Get the pretrained model name for the current task ID
pretrained_model_name="${pretrained_model_names[$SLURM_ARRAY_TASK_ID]}"
echo "Processing model: $pretrained_model_name"

output_dir="${xd_violence_dir}/captions/raw/qwen3-vl-4b/"

# Run the Python script with the specified parameters
python -m src.models.image_captioner \
    --root_path "$root_path" \
    --annotationfile_path "$annotationfile_path" \
    --batch_size "$batch_size" \
    --frame_interval "$frame_interval" \
    --pretrained_model_name "$pretrained_model_name" \
    --output_dir "$output_dir"
