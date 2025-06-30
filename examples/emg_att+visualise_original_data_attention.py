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
    # ... (unchanged, omitted for brevity)
    pass

def load_policy(policy_path: str, dataset_meta, policy_overrides: list = None) -> Tuple[torch.nn.Module, dict]:
    # ... (unchanged, omitted for brevity)
    if policy_overrides:
        overrides = {}
        for override in policy_overrides:
            key, value = override.split('=', 1)
            overrides[key] = value
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
            # ... (unchanged, omitted for brevity)
            continue
        elif key in ["observation.state", "robot_state", "state"]:
            if not isinstance(value, torch.Tensor):
                value = torch.from_numpy(value).type(model_dtype)
            observation[key] = value.unsqueeze(0).to(device)
        elif key == "observation.sensor":
            if not isinstance(value, torch.Tensor):
                value = torch.from_numpy(value).type(model_dtype)
            observation[key] = value.unsqueeze(0).to(device)
    return observation

def analyze_episode(dataset: LeRobotDataset, policy, episode_id: int, device: torch.device,
                   output_dir: str, model_dtype: torch.dtype = torch.float32) -> Dict:
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

    for i in tqdm(range(episode_length), desc="Processing frames"):
        frame = dataset[episode_frames[i]['index'].item()]
        timestamps.append(frame['timestamp'].item())

        # Print observation.sensor if present
        if 'observation.sensor' in frame:
            print(f"Frame {i} observation.sensor: {frame['observation.sensor']}")
        else:
            print(f"Frame {i} observation.sensor: None")

        observation = prepare_observation_for_policy(frame, device, model_dtype, debug=(i==0))

        # Run policy inference
        with torch.inference_mode():
            if hasattr(policy, 'select_action'):
                result = policy.select_action(observation)
                if isinstance(result, tuple):
                    action, attention_maps = result

                    # --- Visualize attention for observation.sensor if present ---
                    if 'observation.sensor' in observation:
                        sensor_attn = policy.get_token_attention(attention_maps, observation, token_key='observation.sensor')
                        print(f"Frame {i} attention for observation.sensor: {sensor_attn}")

                    # Usual visualization
                    visualizations = policy.visualize_attention(
                        attention_maps=attention_maps, 
                        observation=observation,
                    )
                    if attention_videos is None and visualizations:
                        num_cameras = len(visualizations)
                        attention_videos = [[] for _ in range(num_cameras)]
                        print(f"Detected {num_cameras} camera views for attention visualization")
                    if attention_videos is not None:
                        valid_frames_this_step = []
                        for j, vis in enumerate(visualizations):
                            if vis is not None and j < len(attention_videos):
                                attention_videos[j].append(vis.copy())
                                valid_frames_this_step.append(vis.copy())
                            else:
                                valid_frames_this_step.append(None)
                        if len(valid_frames_this_step) == num_cameras and all(f is not None for f in valid_frames_this_step):
                            first_height = valid_frames_this_step[0].shape[0]
                            if all(f.shape[0] == first_height for f in valid_frames_this_step):
                                side_by_side_frame = np.hstack(valid_frames_this_step)
                                side_by_side_buffer.append(side_by_side_frame)
                else:
                    action = result
            else:
                action = policy(observation)

        actions_predicted.append(action.squeeze(0).cpu().numpy())
        if 'action' in frame:
            actions_ground_truth.append(frame['action'].numpy())

    os.makedirs(output_dir, exist_ok=True)
    timestamp_str = time.strftime("%Y%m%d-%H%M%S")

    if attention_videos:
        for i, cam_buffer in enumerate(attention_videos):
            if cam_buffer:
                output_filename = f"{output_dir}/attention_ep{episode_id}_cam{i}_{timestamp_str}.mp4"
                encode_video_ffmpeg(cam_buffer, output_filename, dataset.fps)
        output_filename_sbs = f"{output_dir}/attention_ep{episode_id}_combined_{timestamp_str}.mp4"
        encode_video_ffmpeg(side_by_side_buffer, output_filename_sbs, dataset.fps)

    analysis_results = {
        'episode_id': episode_id,
        'episode_length': episode_length,
        'timestamps': timestamps,
        'actions_predicted': actions_predicted,
        'actions_ground_truth': actions_ground_truth,
    }
    return analysis_results

def main():
    # ... (unchanged, omitted for brevity)
    pass

if __name__ == "__main__":
    main()
