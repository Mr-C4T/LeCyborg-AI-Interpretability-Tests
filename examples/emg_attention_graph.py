import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.ticker import MaxNLocator
from tqdm import tqdm

# === Config ===
input_dir = './emg_viz_results'
fps = 30

# === Style ===
plt.style.use('dark_background')

# === Find CSV files ===
csv_files = [f for f in os.listdir(input_dir) if f.endswith('.csv')]

# === Process each CSV ===
for csv_file in tqdm(csv_files, desc="Processing episodes"):
    csv_path = os.path.join(input_dir, csv_file)
    episode_name = os.path.splitext(csv_file)[0]

    try:
        df = pd.read_csv(csv_path)

        required_cols = {'timestamp', 'emg_attention_raw', 'emg_attention_normalized', 'sensor_value'}
        if not required_cols.issubset(df.columns):
            missing = required_cols - set(df.columns)
            print(f"Skipping {csv_file}: missing columns {missing}")
            continue

        timestamps = df['timestamp'].tolist()
        sensor_values = df['sensor_value'].tolist()
        raw_attention = df['emg_attention_raw'].tolist()
        norm_attention = df['emg_attention_normalized'].tolist()

        # === Averages ===
        sensor_avg = np.mean(sensor_values)
        raw_avg = np.mean(raw_attention)
        norm_avg = np.mean(norm_attention)

        # === Plot Setup ===
        fig, ax_sensor = plt.subplots(figsize=(12, 6))
        ax_sensor.set_facecolor('#121212')
        fig.patch.set_facecolor('#121212')

        # Colors
        color_sensor = "#00FF7F"
        color_raw = "#00BFFF"
        color_norm = "#6638ED"

        # === Left: Sensor Value ===
        ax_sensor.set_xlabel('Timestamp', color='white')
        ax_sensor.set_ylabel('Sensor Value', color=color_sensor)
        ax_sensor.tick_params(axis='y', labelcolor=color_sensor)
        ax_sensor.tick_params(axis='x', colors='white')
        ax_sensor.set_xlim(timestamps[0], timestamps[-1])
        ax_sensor.set_ylim(min(sensor_values) * 0.95, max(sensor_values) * 1.05)
        sensor_line, = ax_sensor.plot([], [], color=color_sensor, linewidth=2, label='Sensor Value')
        ax_sensor.axhline(sensor_avg, color=color_sensor, linestyle='dashed', linewidth=1.5, alpha=0.7)

        # === Right: Raw Attention ===
        ax_raw = ax_sensor.twinx()
        ax_raw.set_ylabel('Raw Attention', color=color_raw)
        ax_raw.tick_params(axis='y', labelcolor=color_raw)
        ax_raw.set_ylim(min(raw_attention) * 0.95, max(raw_attention) * 1.05)
        raw_line, = ax_raw.plot([], [], color=color_raw, linestyle='-', linewidth=2, label='Raw Attention')
        ax_raw.axhline(raw_avg, color=color_raw, linestyle='dashed', linewidth=1.5, alpha=0.7)

        # === Right (offset): Normalized Attention ===
        ax_norm = ax_raw.twinx()
        ax_norm.spines.right.set_position(("outward", 60))  # Offset the second right y-axis
        ax_norm.set_ylabel('Normalized Attention', color=color_norm)
        ax_norm.tick_params(axis='y', labelcolor=color_norm)
        ax_norm.set_ylim(min(norm_attention) * 0.95, max(norm_attention) * 1.05)
        norm_line, = ax_norm.plot([], [], color=color_norm, linestyle='--', linewidth=2, label='Normalized Attention')
        ax_norm.axhline(norm_avg, color=color_norm, linestyle='dashed', linewidth=1.5, alpha=0.7)
        ax_norm.spines['right'].set_color(color_norm)

        # === Title & Legend ===
        plt.title(f'EMG Attention & Sensor — {episode_name}', color='white')
        lines = [sensor_line, raw_line, norm_line]
        labels = [line.get_label() for line in lines]
        ax_sensor.legend(lines, labels, loc='upper left')
        fig.tight_layout()

        # === Animation update ===
        def update(frame_idx):
            x = timestamps[:frame_idx]
            sensor_y = sensor_values[:frame_idx]
            raw_y = raw_attention[:frame_idx]
            norm_y = norm_attention[:frame_idx]
            sensor_line.set_data(x, sensor_y)
            raw_line.set_data(x, raw_y)
            norm_line.set_data(x, norm_y)
            return sensor_line, raw_line, norm_line

        # === Animate & Save ===
        ani = FuncAnimation(fig, update, frames=len(timestamps), interval=1000 / fps, blit=True)
        gif_path = os.path.join(input_dir, f"{episode_name}_dark.gif")
        ani.save(gif_path, writer=PillowWriter(fps=fps))

        # === Save PNG ===
        png_path = os.path.join(input_dir, f"{episode_name}_dark.png")
        plt.savefig(png_path, dpi=300, bbox_inches='tight')
        plt.close(fig)

    except Exception as e:
        print(f"Error processing {csv_file}: {e}")
