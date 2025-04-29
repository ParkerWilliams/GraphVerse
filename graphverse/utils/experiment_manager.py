import csv
import os
import json
from datetime import datetime

def create_experiment_folder(base="experiments"):
    run_name = "run_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(base, run_name)
    os.makedirs(os.path.join(path, "data"), exist_ok=True)
    os.makedirs(os.path.join(path, "evaluation"), exist_ok=True)
    return path

def save_config(config, folder):
    config_path = os.path.join(folder, "config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

def save_error_summary(error_summary, folder):
    path = os.path.join(folder, "evaluation", "error_summary.json")
    with open(path, "w") as f:
        json.dump(error_summary, f, indent=2)

def save_kl_divergence_series(kl_series, folder):
    path = os.path.join(folder, "evaluation", "kl_divergence_timeseries.csv")
    with open(path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["walk_idx", "step_idx", "kl_divergence"])
        for walk_idx, walk_kl in enumerate(kl_series):
            for step_idx, kl in enumerate(walk_kl):
                writer.writerow([walk_idx, step_idx, kl]) 