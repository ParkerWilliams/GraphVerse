import os
import json
import csv
import matplotlib.pyplot as plt
import numpy as np

def plot_error_summary(error_summary_path, output_path=None):
    with open(error_summary_path, "r") as f:
        summary = json.load(f)
    labels = []
    values = []
    for k, v in summary.items():
        if k.endswith("_rate"):
            labels.append(k.replace("_error_rate", ""))
            values.append(v)
    plt.figure(figsize=(8, 5))
    plt.bar(labels, values)
    plt.ylabel("Error Rate")
    plt.title("Model Error Rates")
    plt.ylim(0, 1)
    if output_path:
        plt.savefig(output_path)
    else:
        plt.show()
    plt.close()

def plot_kl_divergence_timeseries(kl_csv_path, walk_idx=0, output_path=None, zoom_steps=None):
    # Read KL series for all walks
    walk_kl = []
    with open(kl_csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            idx = int(row["walk_idx"])
            if idx == walk_idx:
                walk_kl.append((int(row["step_idx"]), float(row["kl_divergence"])))
    if not walk_kl:
        print(f"No KL data for walk {walk_idx}")
        return
    walk_kl.sort()
    steps, kls = zip(*walk_kl)
    if zoom_steps is not None:
        steps = steps[-zoom_steps:]
        kls = kls[-zoom_steps:]
    plt.figure(figsize=(10, 4))
    plt.plot(steps, kls, marker="o")
    plt.xlabel("Step")
    plt.ylabel("KL Divergence")
    plt.title(f"KL Divergence Time Series (Walk {walk_idx})")
    if output_path:
        plt.savefig(output_path)
    else:
        plt.show()
    plt.close()

def plot_aggregate_kl(kl_csv_path, output_path=None):
    # Aggregate KL by step across all walks
    step_kl = {}
    with open(kl_csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            step = int(row["step_idx"])
            kl = float(row["kl_divergence"])
            step_kl.setdefault(step, []).append(kl)
    steps = sorted(step_kl.keys())
    means = [np.mean(step_kl[s]) for s in steps]
    stds = [np.std(step_kl[s]) for s in steps]
    plt.figure(figsize=(10, 4))
    plt.plot(steps, means, label="Mean KL")
    plt.fill_between(steps, np.array(means)-np.array(stds), np.array(means)+np.array(stds), alpha=0.3, label="Std Dev")
    plt.xlabel("Step")
    plt.ylabel("KL Divergence")
    plt.title("Aggregate KL Divergence Across Walks")
    plt.legend()
    if output_path:
        plt.savefig(output_path)
    else:
        plt.show()
    plt.close()