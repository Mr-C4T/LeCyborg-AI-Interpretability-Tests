#!/usr/bin/env python

"""
Script to analyze policy behavior on dataset episodes.
Runs policy inference on episodes and analyzes feature importance, including sensor attention.
By default, analyzes all episodes in the dataset.
"""

import argparse
import os
import time
import subprocess
from typing import Dict, Tuple

import numpy as np
import torch
from tqdm import tqdm

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.policies.factory import make_policy
from lerobot.configs.policies import PreTrainedConfig

from src.attention_maps import ACTPolicyWithAttention

def none_or_int(value):
    if value == "None":
        return None
    return int(value)

def encode_video_ffmpeg(frames, output_filename, fps, pix_fmt_in="bgr24"):
    if not frames:
        print(f"No frames to encode for {output_filename}.")
        return

    height, width, channels = frames[0].shape
    if channels != 3:
        print(f"Error: Frames must be 3-channel (BGR). Got {channels} channels.")
        return

    command = [
        'ffmpeg', '-y', '-f', 'rawvideo', '-vcodec', 'rawvideo',
        '-s', f'{width}x{height}', '-pix_fmt', pix_fmt_in,
        '-r', str(fps), '-i', '-', '-an', '-vcodec', 'libx264',
        '-pix_fmt', 'yuv420p', '-crf', '23', output_filename
    ]

    try:
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        for frame in frames:
            process.stdin.write(frame.tobytes())
        stdout, stderr = process.communicate()
        if process.returncode != 0:
            print(f"Error encoding video {output_filename}:")
            print(f"FFmpeg stderr:\n{stderr.decode(errors='ignore')}")
        else:
            print(f"Successfully encoded video: {output_filename}")
    except FileNotFoundError:
        print("Error: ffmpeg command not found. Please ensure ffmpeg is installed and in your PATH.")
    except Exception as e:
        print(f"Unexpected error during video encoding for {output_filename}: {e}")

def load_policy(policy_path: str, dataset_meta, policy_overrides: list = None) -> Tuple[torch.nn.Module, dict]:
    if policy_overrides:
        overrides = dict(override.split('=', 1) for override in policy_overrides)
        policy_cfg = PreTrainedConfig.from_pretrained(policy_path, **overrides)
    else:
        policy_cfg = PreTrainedConfig.from_pretrained(policy_path)
        policy_cfg.pretrained_path = policy_path

    policy = make_policy(policy_cfg, ds_meta=dataset_meta)
    policy = ACTPolicyWithAttention(policy)
    return policy, policy_cfg

def prepare_observation_for_policy(frame: dict, device: torch.device, model_dtype: torch.dtype = torch.float32, debug: bool = False) -> dict:
    observation = {}
    for key, value in frame.items():
        if "image" in key:
            if isinstance(value, torch.Tensor):
                while value.dim() > 3:
                    value = value.squeeze(0)
                if value.dim() == 3:
                    value = value.permute(2, 0, 1) if value.shape[2] in [1, 3] else value
                    value = value.type(model_dtype) / 255.0 if value.max() > 1.0 else value
                observation[key] = value.unsqueeze(0).to(device)
        elif key in ["observation.state", "robot_state", "state", "observation.sensor"]:
            if not isinstance(value, torch.Tensor):
                value = torch.tensor(value, dtype=model_dtype)
            observation[key] = value.unsqueeze(0).to(device)
    return observation

def analyze_episode(dataset: LeRobotDataset, policy, episode_id: int, device: torch.device, output_dir: str, model_dtype: torch.dtype = torch.float32) -> Dict:
    episode_frames = dataset.hf_dataset.filter(lambda x: x["episode_index"] == episode_id)
    episode_length = len(episode_frames)
    if episode_length == 0:
        raise ValueError(f"Episode {episode_id} not found or is empty")
    print(f"Analyzing episode {episode_id} with {episode_length} frames")

    attention_videos = None
    side_by_side_buffer = []
    actions_predicted = []
    actions_ground_truth = []
    timestamps = []
    emg_attention_raw_values = []
    emg_attention_norm_values = []
    sensor_values = []

    for i in tqdm(range(episode_length), desc="Processing frames"):
        frame = dataset[episode_frames[i]['index'].item()]
        timestamps.append(frame['timestamp'].item())

        sensor_val = frame.get("observation.sensor")
        if isinstance(sensor_val, torch.Tensor):
            sensor_val = sensor_val.item()
        sensor_values.append(sensor_val)

        observation = prepare_observation_for_policy(frame, device, model_dtype, debug=(i == 0))

        with torch.inference_mode():
            result = policy.select_action(observation) if hasattr(policy, 'select_action') else policy(observation)
            if isinstance(result, tuple):
                action, attention_maps = result
                attn = getattr(policy, "last_raw_attention", None)
                sensor_idx = policy.token_key_to_index.get('observation.sensor', None)
                proprio_idx = policy.token_key_to_index.get('observation.state', None)
                if attn is not None and sensor_idx is not None:
                    while attn.dim() > 4:
                        attn = attn[0]
                    attn_avg = attn.mean(dim=(0, 1)) if attn.dim() == 4 else attn[0] if attn.dim() == 3 else None
                    if attn_avg is not None and sensor_idx < attn_avg.shape[1]:
                        emg_att_raw = attn_avg[:, sensor_idx].mean().item()
                        attention_values = attn_avg.cpu().numpy().flatten().tolist()
                        attention_values.append(emg_att_raw)
                        global_min = min(attention_values)
                        global_max = max(attention_values)
                        emg_att_norm = (emg_att_raw - global_min) / (global_max - global_min) if global_max > global_min else 0.0
                        emg_attention_raw_values.append(emg_att_raw)
                        emg_attention_norm_values.append(emg_att_norm)
                    else:
                        emg_attention_raw_values.append(None)
                        emg_attention_norm_values.append(None)
                else:
                    emg_attention_raw_values.append(None)
                    emg_attention_norm_values.append(None)
                visualizations = policy.visualize_attention(attention_maps=attention_maps, observation=observation)
                if attention_videos is None and visualizations:
                    num_cameras = len(visualizations)
                    attention_videos = [[] for _ in range(num_cameras)]
                if attention_videos is not None:
                    valid_frames_this_step = []
                    for j, vis in enumerate(visualizations):
                        if vis is not None and j < len(attention_videos):
                            attention_videos[j].append(vis.copy())
                            valid_frames_this_step.append(vis.copy())
                        else:
                            valid_frames_this_step.append(None)
                    if all(f is not None for f in valid_frames_this_step):
                        side_by_side_frame = np.hstack(valid_frames_this_step)
                        side_by_side_buffer.append(side_by_side_frame)
            else:
                action = result

        actions_predicted.append(action.squeeze(0).cpu().numpy())
        if 'action' in frame:
            actions_ground_truth.append(frame['action'].numpy())

    os.makedirs(output_dir, exist_ok=True)
    timestamp_str = time.strftime("%Y%m%d-%H%M%S")

    if attention_videos:
        for i, cam_buffer in enumerate(attention_videos):
            if cam_buffer:
                encode_video_ffmpeg(cam_buffer, f"{output_dir}/attention_ep{episode_id}_cam{i}_{timestamp_str}.mp4", dataset.fps)
        encode_video_ffmpeg(side_by_side_buffer, f"{output_dir}/attention_ep{episode_id}_combined_{timestamp_str}.mp4", dataset.fps)

    analysis_results = {
        'episode_id': episode_id,
        'episode_length': episode_length,
        'timestamps': timestamps,
        'actions_predicted': actions_predicted,
        'actions_ground_truth': actions_ground_truth,
        'emg_attention_raw': emg_attention_raw_values,
        'emg_attention_normalized': emg_attention_norm_values,
        'sensor_values': sensor_values,
    }
    return analysis_results

def main():
    parser = argparse.ArgumentParser(description="Analyze policy behavior on dataset episodes")
    parser.add_argument("--dataset-repo-id", type=str, required=True)
    parser.add_argument("--episode-id", type=int, default=None)
    parser.add_argument("--policy-path", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="./analysis_output")
    parser.add_argument("--policy-overrides", type=str, nargs="*")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--model-dtype", type=str, default="float32", choices=["float32", "float16", "bfloat16"])
    args = parser.parse_args()

    device = torch.device(args.device)
    model_dtype = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[args.model_dtype]

    dataset = LeRobotDataset(args.dataset_repo_id)
    episodes_to_analyze = [args.episode_id] if args.episode_id is not None else list(range(dataset.num_episodes))

    policy, policy_cfg = load_policy(args.policy_path, dataset.meta, args.policy_overrides)
    if hasattr(policy, 'model'):
        policy.model.eval().to(device)
    elif hasattr(policy, 'eval'):
        policy.eval()

    all_results = []
    for episode_id in tqdm(episodes_to_analyze, desc="Analyzing episodes"):
        try:
            results = analyze_episode(dataset, policy, episode_id, device, args.output_dir, model_dtype)
            csv_path = os.path.join(args.output_dir, f"emg_attention_ep{episode_id}.csv")
            import csv
            with open(csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['timestamp', 'emg_attention_raw', 'emg_attention_normalized', 'sensor_value'])
                for t, raw, norm, sensor in zip(results['timestamps'], results['emg_attention_raw'], results['emg_attention_normalized'], results['sensor_values']):
                    writer.writerow([t, raw, norm, sensor])
            all_results.append(results)
        except Exception as e:
            print(f"Failed to analyze episode {episode_id}: {e}")
            import traceback
            traceback.print_exc()

    print(f"Finished analysis. Results saved to: {args.output_dir}")

if __name__ == "__main__":
    main()
