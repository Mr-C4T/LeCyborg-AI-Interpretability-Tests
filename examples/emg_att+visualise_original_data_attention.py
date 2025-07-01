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
    """
    Encodes a list of numpy frames into a video using ffmpeg.
    """
    if not frames:
        print(f"No frames to encode for {output_filename}.")
        return

    height, width, channels = frames[0].shape
    if channels != 3:
        print(f"Error: Frames must be 3-channel (BGR). Got {channels} channels.")
        return

    command = [
        'ffmpeg',
        '-y',  # Overwrite output file if it exists
        '-f', 'rawvideo',
        '-vcodec', 'rawvideo',
        '-s', f'{width}x{height}',  # Frame size
        '-pix_fmt', pix_fmt_in,     # Input pixel format
        '-r', str(fps),             # Frames per second
        '-i', '-',                  # Input comes from stdin
        '-an',                      # No audio
        '-vcodec', 'libx264',       # Output video codec
        '-pix_fmt', 'yuv420p',      # Output pixel format for broad compatibility
        '-crf', '23',               # Constant Rate Factor (quality, 18-28 is good range)
        output_filename
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
        print(f"An unexpected error occurred during video encoding for {output_filename}: {e}")

def load_policy(policy_path: str, dataset_meta, policy_overrides: list = None) -> Tuple[torch.nn.Module, dict]:
    """Load and initialize a policy from checkpoint."""
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
    """Convert dataset frame to policy observation format."""
    observation = {}
    for key, value in frame.items():
        if "image" in key:
            if debug:
                print(f"Processing {key}: original shape {value.shape}, dtype {value.dtype}")
            # Convert image to policy format: channel first, float32 in [0,1], with batch dimension
            if isinstance(value, torch.Tensor):
                while value.dim() > 3:
                    value = value.squeeze(0)
                if value.dim() != 3:
                    raise ValueError(f"Expected 3D tensor for {key} after squeezing, got shape {value.shape}")
                h, w, c = value.shape
                if c not in [1, 3]:
                    if h in [1, 3]:
                        pass  # Format is (C, H, W) - already correct
                    elif w in [1, 3]:
                        value = value.permute(2, 0, 1)  # (H, W, C) -> (C, H, W)
                    else:
                        value = value.permute(2, 0, 1)
                else:
                    value = value.permute(2, 0, 1)
                if debug:
                    print(f"After permutation: {value.shape}")
                if value.dtype != model_dtype:
                    value = value.type(model_dtype)
                if value.max() > 1.0:
                    value = value / 255.0
                if debug:
                    print(f"Final shape for {key}: {value.shape}, range: [{value.min():.3f}, {value.max():.3f}]")
            observation[key] = value.unsqueeze(0).to(device)  # Add batch dimension
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
    """
    Run policy inference on an episode and analyze feature importance, including sensor attention.
    Returns:
        Dictionary containing analysis results
    """
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

    for i in tqdm(range(episode_length), desc="Processing frames"):
        frame = dataset[episode_frames[i]['index'].item()]
        timestamps.append(frame['timestamp'].item())

        if 'observation.sensor' in frame:
            print(f"Frame {i} observation.sensor: {frame['observation.sensor']}")
        else:
            print(f"Frame {i} observation.sensor: None")

        observation = prepare_observation_for_policy(frame, device, model_dtype, debug=(i == 0))

        with torch.inference_mode():
            if hasattr(policy, 'select_action'):
                result = policy.select_action(observation)
                if isinstance(result, tuple):
                    action, attention_maps = result

                    # --- Sensor Attention Analysis ---
                    attn = getattr(policy, "last_raw_attention", None)
                    sensor_idx = policy.token_key_to_index.get('observation.sensor', None)
                    proprio_idx = policy.token_key_to_index.get('observation.state', None)

                    if attn is not None and sensor_idx is not None:
                        # Remove batch dim
                        while attn.dim() > 4:
                            attn = attn[0]
                        if attn.dim() == 4:
                            attn_avg = attn.mean(dim=(0, 1))
                        elif attn.dim() == 3:
                            attn_avg = attn[0]
                        else:
                            attn_avg = None

                        if attn_avg is not None and sensor_idx < attn_avg.shape[1]:
                            emg_att_raw = attn_avg[:, sensor_idx].mean().item()
                            attention_values = []

                            if proprio_idx is not None and proprio_idx < attn_avg.shape[1]:
                                proprio_att = attn_avg[:, proprio_idx].mean().item()
                                attention_values.append(proprio_att)

                            attention_values.extend(attn_avg.cpu().numpy().flatten())
                            attention_values.append(emg_att_raw)

                            global_min = float(np.min(attention_values))
                            global_max = float(np.max(attention_values))
                            emg_att_norm = (emg_att_raw - global_min) / (global_max - global_min) if global_max > global_min else 0.0

                            print(f"Frame {i} raw attention for observation.sensor (EMG): {emg_att_raw:.6f}")
                            print(f"Frame {i} normalized attention for observation.sensor (EMG): {emg_att_norm:.6f}")

                            emg_attention_raw_values.append(emg_att_raw)
                            emg_attention_norm_values.append(emg_att_norm)
                        else:
                            print(f"Frame {i} sensor index out of bounds or attention missing")
                            emg_attention_raw_values.append(None)
                            emg_attention_norm_values.append(None)
                    else:
                        print(f"Frame {i} no raw attention available or sensor token missing")
                        emg_attention_raw_values.append(None)
                        emg_attention_norm_values.append(None)

                    # --- Visualization ---
                    visualizations = policy.visualize_attention(attention_maps=attention_maps, observation=observation)
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
                            if all(f.shape[0] == valid_frames_this_step[0].shape[0] for f in valid_frames_this_step):
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
        'emg_attention_raw': emg_attention_raw_values,
        'emg_attention_normalized': emg_attention_norm_values,
    }
    return analysis_results


def main():
    parser = argparse.ArgumentParser(description="Analyze policy behavior on dataset episodes")
    parser.add_argument("--dataset-repo-id", type=str, required=True,
                        help="Repository ID of the dataset to analyze")
    parser.add_argument("--episode-id", type=int, default=None,
                        help="Episode ID to analyze (if not specified, analyzes all episodes)")
    parser.add_argument("--policy-path", type=str, required=True,
                        help="Path to the policy checkpoint")
    parser.add_argument("--output-dir", type=str, default="./analysis_output",
                        help="Directory to save analysis results")
    parser.add_argument("--policy-overrides", type=str, nargs="*",
                        help="Policy config overrides in key=value format")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use for inference")
    parser.add_argument("--model-dtype", type=str, default="float32",
                        choices=["float32", "float16", "bfloat16"],
                        help="Model data type")
    
    args = parser.parse_args()
    
    # Set up device and dtype
    device = torch.device(args.device)
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16, 
        "bfloat16": torch.bfloat16
    }
    model_dtype = dtype_map[args.model_dtype]
    
    print(f"Loading dataset: {args.dataset_repo_id}")
    print(f"Policy path: {args.policy_path}")
    print(f"Using device: {device}")
    
    # Load dataset
    try:
        dataset = LeRobotDataset(args.dataset_repo_id)
        print(f"Dataset loaded successfully. Total episodes: {dataset.num_episodes}")
        
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return
    
    # Determine which episodes to analyze
    if args.episode_id is not None:
        # Single episode analysis
        if args.episode_id >= dataset.num_episodes:
            raise ValueError(f"Episode {args.episode_id} not found. Dataset has {dataset.num_episodes} episodes.")
        episodes_to_analyze = [args.episode_id]
        print(f"Target episode: {args.episode_id}")
    else:
        # All episodes analysis
        episodes_to_analyze = list(range(dataset.num_episodes))
        print(f"Will analyze all {dataset.num_episodes} episodes")
    
    # Load policy
    try:
        print("Loading policy...")
        policy, policy_cfg = load_policy(
            args.policy_path,
            dataset.meta,
            args.policy_overrides
        )
        
        if hasattr(policy, 'model'):
            policy.model.eval()
            policy.model.to(device)
        elif hasattr(policy, 'eval'):
            policy.eval()
            
        print("Policy loaded successfully")
        
    except Exception as e:
        print(f"Error loading policy: {e}")
        return
    
    # Run analysis on all specified episodes
    all_results = []
    failed_episodes = []
    
    for episode_id in tqdm(episodes_to_analyze, desc="Analyzing episodes"):
        try:
            print(f"\nStarting analysis of episode {episode_id}...")
            results = analyze_episode(
                dataset=dataset,
                policy=policy,
                episode_id=episode_id,
                device=device,
                output_dir=args.output_dir,
                model_dtype=model_dtype
            )

            # Optional: Save EMG attention values to CSV
            import csv
            emg_csv_path = os.path.join(args.output_dir, f"emg_attention_ep{episode_id}.csv")
            with open(emg_csv_path, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['timestamp', 'emg_attention_raw', 'emg_attention_normalized'])
                for t, raw, norm in zip(results['timestamps'], results['emg_attention_raw'], results['emg_attention_normalized']):
                    writer.writerow([t, raw, norm])
            all_results.append(results)
            print(f"Episode {episode_id} analysis completed successfully")
            
        except Exception as e:
            print(f"Error analyzing episode {episode_id}: {e}")
            failed_episodes.append(episode_id)
            import traceback
            traceback.print_exc()
            continue
    
    # Summary
    print(f"\n{'='*60}")
    print(f"ANALYSIS SUMMARY")
    print(f"{'='*60}")
    print(f"Successfully analyzed: {len(all_results)} episodes")
    if failed_episodes:
        print(f"Failed episodes: {len(failed_episodes)} ({failed_episodes})")
    else:
        print("No failed episodes")
    print(f"Results saved to: {args.output_dir}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
