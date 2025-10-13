import json
import numpy as np
import os
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def analyze_and_plot():
    # --- Data Preparation ---
    # The script is inside 'cameraready', so the relative path to results needs to be adjusted.
    lora_dir = "../results/results_sequencial/composition/lora"
    files_by_gap = {
        0: [
            "250509_185748_lora_llava_port1_seqgap0_testnum500.json", "250930_200315_lora_llava_port1_seqgap0_testnum500.json",
            "250930_200452_lora_llava_port1_seqgap0_testnum500.json", "250930_200519_lora_llava_port1_seqgap0_testnum500.json",
            "250930_200539_lora_llava_port1_seqgap0_testnum500.json"
        ],
        10: [
            "250509_232824_lora_llava_port1_seqgap10_testnum500.json", "250930_233441_lora_llava_port1_seqgap10_testnum500.json",
            "250930_233644_lora_llava_port1_seqgap10_testnum500.json", "250930_233657_lora_llava_port1_seqgap10_testnum500.json",
            "250930_233834_lora_llava_port1_seqgap10_testnum500.json"
        ],
        20: [
            "250510_040108_lora_llava_port1_seqgap20_testnum500.json", "251001_030925_lora_llava_port1_seqgap20_testnum500.json",
            "251001_031200_lora_llava_port1_seqgap20_testnum500.json", "251001_031209_lora_llava_port1_seqgap20_testnum500.json",
            "251001_031613_lora_llava_port1_seqgap20_testnum500.json"
        ],
        50: [
            "250510_084333_lora_llava_port1_seqgap50_testnum500.json", "251001_064739_lora_llava_port1_seqgap50_testnum500.json",
            "251001_065035_lora_llava_port1_seqgap50_testnum500.json", "251001_065036_lora_llava_port1_seqgap50_testnum500.json",
            "251001_065720_lora_llava_port1_seqgap50_testnum500.json"
        ],
        100: [
            "250510_133202_lora_llava_port1_seqgap100_testnum500.json",
            "251001_103559_lora_llava_port1_seqgap100_testnum500.json", "251001_103917_lora_llava_port1_seqgap100_testnum500.json",
            "251001_103932_lora_llava_port1_seqgap100_testnum500.json", "251001_104833_lora_llava_port1_seqgap100_testnum500.json"
        ]
    }
    metrics_to_track = [
        "vis/inner/acc_val", "vis/edit/acc_val", "vis/image_rephrase/acc_val",
        "vis/loc/acc_val", "vis/image_loc/acc_val", "text/inner/acc_val",
        "text/edit/acc_val", "text/loc/acc_val", "port/acc_val"
    ]

    # --- Data Aggregation ---
    plot_data = defaultdict(lambda: defaultdict(list))
    for gap, files in sorted(files_by_gap.items()):
        for filename in files:
            filepath = os.path.join(lora_dir, filename)
            try:
                with open(filepath, "r") as f:
                    data = json.load(f)
                    for metric in metrics_to_track:
                        value = data.get("results", {}).get(metric)
                        if value is not None:
                            plot_data[metric]["gap"].append(gap)
                            plot_data[metric]["value"].append(value)
            except (FileNotFoundError, json.JSONDecodeError) as e:
                print(f"Warning: Could not read or parse {filepath}: {e}")


    # --- Plotting and Data Collection for CSV ---
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(3, 3, figsize=(20, 18))
    fig.suptitle("LoRA Performance vs. Sequential Edit Gap (testnum=500)", fontsize=24, y=0.98)
    axes = axes.flatten()
    
    summary_data = []

    for i, metric in enumerate(metrics_to_track):
        ax = axes[i]
        
        if not plot_data[metric]["value"]:
            ax.set_title(f"{metric}\\n(No data)", fontsize=12)
            ax.set_ylim(0, 1)
            continue

        gaps = plot_data[metric]["gap"]
        values = plot_data[metric]["value"]
        
        unique_gaps = sorted(list(set(gaps)))
        means = [np.mean([v for g, v in zip(gaps, values) if g == ug]) for ug in unique_gaps]
        stds = [np.std([v for g, v in zip(gaps, values) if g == ug]) for ug in unique_gaps]

        # Collect data for CSV
        for ug, mean, std in zip(unique_gaps, means, stds):
            summary_data.append({"metric": metric, "gap": ug, "mean": mean, "std": std})

        ax.plot(unique_gaps, means, marker="o", linestyle="-", label="Mean")
        ax.fill_between(unique_gaps, np.array(means) - np.array(stds), np.array(means) + np.array(stds), alpha=0.2, label="Std Dev")

        metric_name_short = metric.replace("_val", "").replace("/", " / ")
        ax.set_title(metric_name_short, fontsize=14)
        ax.set_xlabel("Gap", fontsize=10)
        ax.set_ylabel("Accuracy", fontsize=10)
        ax.set_xticks(unique_gaps)
        ax.set_ylim(0, 1.0)
        ax.grid(True, which="both", linestyle="--", linewidth=0.5)
        ax.legend()

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    output_filename = "lora_perf_vs_gap_all_metrics.png"
    plt.savefig(output_filename)
    plt.close()
    print(f"Generated combined plot: {output_filename}")

    # --- Save results to CSV ---
    if summary_data:
        df = pd.DataFrame(summary_data)
        csv_filename = "lora_performance_summary.csv"
        df.to_csv(csv_filename, index=False, float_format="%.4f")
        print(f"Saved summary statistics to: {csv_filename}")

if __name__ == "__main__":
    analyze_and_plot()
