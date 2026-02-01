import argparse
import json
import os
from pathlib import Path

import torch
from src.utils.path_utils import find_unprocessed_videos
from tqdm import tqdm

from src.data.video_record import VideoRecord
from src.model.qwen3_vl_client import Qwen3VLClient

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'


class ImageCaptioner:
    def __init__(
        self,
        batch_size,
        frame_interval,
        imagefile_template,
        pretrained_model_name,
        dtype_str,
        output_dir,
    ):
        self.batch_size = batch_size
        self.frame_interval = frame_interval
        self.imagefile_template = imagefile_template
        self.dtype = self._get_dtype(dtype_str)
        self.qwen = Qwen3VLClient(model_path=pretrained_model_name, dtype_str=dtype_str)
        self.output_dir = Path(output_dir)

    def _get_dtype(self, dtype_str):
        if dtype_str == "bfloat16":
            return torch.bfloat16
        return torch.float16 if dtype_str == "float16" else torch.float32

    def _caption_image(self, image_path):
        return self.qwen.generate_caption(image_path)

    def process_video(self, video):
        print(video.path)
        video_name = os.path.basename(video.path)
        video_captions = {}

        for batch_start_frame in tqdm(
            range(0, video.num_frames, self.batch_size * self.frame_interval),
            desc=f"Processing {video.path}",
            unit="batch",
        ):
            batch_end_frame = min(
                batch_start_frame + (self.batch_size * self.frame_interval), video.num_frames
            )
            batch_frame_idxs = range(batch_start_frame, batch_end_frame, self.frame_interval)
            #print(batch_frame_idxs)
            #"""
            batch_frame_paths = [
                Path(video.path) / self.imagefile_template.format(frame_idx)
                for frame_idx in batch_frame_idxs
            ]
            batch_generated_text = self.qwen.generate_caption_batch(
                [str(frame_path) for frame_path in batch_frame_paths]
            )

            for frame_idx, generated_text in zip(batch_frame_idxs, batch_generated_text):
                generated_text = generated_text.strip()
                video_captions[frame_idx] = generated_text
                print(video_captions[frame_idx])

        output_path = self.output_dir / f"{video_name}.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(video_captions, f, indent=4,ensure_ascii=False)


def run(
    root_path,
    annotationfile_path,
    batch_size,
    frame_interval,
    imagefile_template,
    pretrained_model_name,
    output_dir,
    dtype,
    resume,
    pathname,
):
    captioner = ImageCaptioner(
        batch_size,
        frame_interval,
        imagefile_template,
        pretrained_model_name,
        dtype,
        output_dir,
    )

    video_list = [VideoRecord(x.strip().split(), root_path) for x in open(annotationfile_path)]
    if resume:
        video_list = find_unprocessed_videos(video_list, captioner.output_dir, pathname)

    for video in video_list:
        captioner.process_video(video)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_path", type=str, required=True)
    parser.add_argument("--annotationfile_path", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--frame_interval", type=int, default=16)
    parser.add_argument("--imagefile_template", type=str, default="{:06d}.jpg")
    parser.add_argument(
        "--pretrained_model_name",
        type=str,
        default=None,
        help="Local Qwen3-VL-4B path. If not provided, uses MODEL_PATH env var.",
    )
    # parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default=None)  # 先设为 None
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["bfloat16", "float16", "float32"],
        default="bfloat16",
        help="Data type (bfloat16, float16 or float32)",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--pathname", type=str, default="*.json")
    args = parser.parse_args()
    if args.output_dir is None:
        args.output_dir = f"{args.root_path}/../captions/raw/{os.path.basename(args.pretrained_model_name)}/"
    return args


if __name__ == "__main__":
    args = parse_args()
    run(
        args.root_path,
        args.annotationfile_path,
        args.batch_size,
        args.frame_interval,
        args.imagefile_template,
        args.pretrained_model_name,
        args.output_dir,
        args.dtype,
        args.resume,
        args.pathname,
    )
