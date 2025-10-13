import json
import numpy as np
import os
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def analyze_and_plot():
    # --- Data Preparation ---
    # The script is inside 'cameraready', so paths are relative to the project root.
    output_dir = "cameraready"
    
    all_files = {
        "lora": {
            "dir": "results/results_sequencial/minigpt4/composition/lora",
            "files_by_gap": {
                0: [
                    "250507_205923_lora_minigpt4_port1_seqgap0_testnum500.json",
                    "251001_183950_lora_minigpt4_port1_seqgap0_testnum500.json",
                    "251001_184006_lora_minigpt4_port1_seqgap0_testnum500.json",
                    "251001_184618_lora_minigpt4_port1_seqgap0_testnum500.json",
                    "251001_184631_lora_minigpt4_port1_seqgap0_testnum500.json"
                ],
                10: [
                    "250507_230239_lora_minigpt4_port1_seqgap10_testnum500.json",
                    "251001_204139_lora_minigpt4_port1_seqgap10_testnum500.json",
                    "251001_204314_lora_minigpt4_port1_seqgap10_testnum500.json",
                    "251001_204750_lora_minigpt4_port1_seqgap10_testnum500.json",
                    "251001_204957_lora_minigpt4_port1_seqgap10_testnum500.json"
                ],
                20: [
                    "250508_010802_lora_minigpt4_port1_seqgap20_testnum500.json",
                    "251001_224554_lora_minigpt4_port1_seqgap20_testnum500.json",
                    "251001_224822_lora_minigpt4_port1_seqgap20_testnum500.json",
                    "251001_225335_lora_minigpt4_port1_seqgap20_testnum500.json",
                    "251001_225518_lora_minigpt4_port1_seqgap20_testnum500.json"
                ],
                50: [
                    "250508_031551_lora_minigpt4_port1_seqgap50_testnum500.json",
                    "251002_005150_lora_minigpt4_port1_seqgap50_testnum500.json",
                    "251002_005605_lora_minigpt4_port1_seqgap50_testnum500.json",
                    "251002_010133_lora_minigpt4_port1_seqgap50_testnum500.json",
                    "251002_010250_lora_minigpt4_port1_seqgap50_testnum500.json"
                ],
                100: [
                    "250508_052933_lora_minigpt4_port1_seqgap100_testnum500.json",
                    "251002_030449_lora_minigpt4_port1_seqgap100_testnum500.json",
                    "251002_031035_lora_minigpt4_port1_seqgap100_testnum500.json",
                    "251002_031651_lora_minigpt4_port1_seqgap100_testnum500.json",
                    "251002_031747_lora_minigpt4_port1_seqgap100_testnum500.json"
                ]
            }
        },
        "ft": {
            "dir": "results/results_sequencial/minigpt4/composition/ft",
            "files_by_gap": {
                0: [
                    "250507_205854_ft_minigpt4_port1_seqgap0_testnum500.json",
                    "251002_111733_ft_minigpt4_port1_seqgap0_testnum500.json",
                    "251002_111749_ft_minigpt4_port1_seqgap0_testnum500.json",
                    "251002_111817_ft_minigpt4_port1_seqgap0_testnum500.json",
                    "251002_111826_ft_minigpt4_port1_seqgap0_testnum500.json"
                ],
                10: [
                    "250507_214419_ft_minigpt4_port1_seqgap10_testnum500.json",
                    "251002_120312_ft_minigpt4_port1_seqgap10_testnum500.json",
                    "251002_120335_ft_minigpt4_port1_seqgap10_testnum500.json",
                    "251002_120337_ft_minigpt4_port1_seqgap10_testnum500.json",
                    "251002_120419_ft_minigpt4_port1_seqgap10_testnum500.json"
                ],
                20: [
                    "250507_223016_ft_minigpt4_port1_seqgap20_testnum500.json",
                    "251002_124922_ft_minigpt4_port1_seqgap20_testnum500.json",
                    "251002_124923_ft_minigpt4_port1_seqgap20_testnum500.json",
                    "251002_124949_ft_minigpt4_port1_seqgap20_testnum500.json",
                    "251002_125044_ft_minigpt4_port1_seqgap20_testnum500.json"
                ],
                50: [
                    "250507_231654_ft_minigpt4_port1_seqgap50_testnum500.json",
                    "251002_133541_ft_minigpt4_port1_seqgap50_testnum500.json",
                    "251002_133619_ft_minigpt4_port1_seqgap50_testnum500.json",
                    "251002_133636_ft_minigpt4_port1_seqgap50_testnum500.json",
                    "251002_133747_ft_minigpt4_port1_seqgap50_testnum500.json"
                ],
                100: [
                    "250508_000518_ft_minigpt4_port1_seqgap100_testnum500.json",
                    "251002_142401_ft_minigpt4_port1_seqgap100_testnum500.json",
                    "251002_142511_ft_minigpt4_port1_seqgap100_testnum500.json",
                    "251002_142530_ft_minigpt4_port1_seqgap100_testnum500.json",
                    "251002_142651_ft_minigpt4_port1_seqgap100_testnum500.json"
                ]
            }
        }
    }

    metrics_to_track = [
        "vis/inner/acc_val", "text/inner/acc_val", "port/acc_val",
        "vis/edit/acc_val", "vis/image_rephrase/acc_val",
        "vis/loc/acc_val", "vis/image_loc/acc_val", "text/edit/acc_val",
        "text/loc/acc_val"
    ]
    
    metric_titles = {
        "vis/inner/acc_val": "Vis Rel",
        "text/inner/acc_val": "Text Rel",
        "port/acc_val": "Comp Rel"
    }


    # --- Data Aggregation ---
    plot_data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for model_name, model_data in all_files.items():
        base_dir = model_data["dir"]
        for gap, files in sorted(model_data["files_by_gap"].items()):
            for filename in files:
                filepath = os.path.join(base_dir, filename)
                try:
                    with open(filepath, "r") as f:
                        data = json.load(f)
                        for metric in metrics_to_track:
                            value = data.get("results", {}).get(metric)
                            if value is not None:
                                plot_data[model_name][metric]["gap"].append(gap)
                                plot_data[model_name][metric]["value"].append(value)
                except (FileNotFoundError, json.JSONDecodeError) as e:
                    print(f"Warning: Could not read or parse {filepath}: {e}")

    # --- Plotting and Data Collection for CSV ---
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(3, 3, figsize=(20, 18))
    fig.suptitle("MiniGPT4 Performance vs. Sequential Edit Gap (testnum=500)", fontsize=24, y=0.98)
    axes = axes.flatten()
    
    all_summary_data = defaultdict(list)
    model_colors = {
        "lora": "skyblue",
        "ft": "purple"
    }

    for i, metric in enumerate(metrics_to_track):
        ax = axes[i]
        
        has_data = any(plot_data[model][metric]["value"] for model in all_files)
        if not has_data:
            ax.set_title(f"{metric}\n(No data)", fontsize=12)
            ax.set_ylim(0, 1)
            continue

        for model_name in all_files.keys():
            if not plot_data[model_name][metric]["value"]:
                continue

            gaps = plot_data[model_name][metric]["gap"]
            values = plot_data[model_name][metric]["value"]
            
            unique_gaps = sorted(list(set(gaps)))
            means = [np.mean([v for g, v in zip(gaps, values) if g == ug]) for ug in unique_gaps]
            stds = [np.std([v for g, v in zip(gaps, values) if g == ug]) for ug in unique_gaps]

            # Collect data for CSV
            for ug, mean, std in zip(unique_gaps, means, stds):
                all_summary_data[model_name].append({"metric": metric, "gap": ug, "mean": mean, "std": std})

            color = model_colors.get(model_name, "black")
            ax.plot(unique_gaps, means, marker="o", linestyle="-", label=f"{model_name.upper()} Mean", color=color)
            ax.fill_between(unique_gaps, np.array(means) - np.array(stds), np.array(means) + np.array(stds), alpha=0.2, color=color)

        metric_name_short = metric_titles.get(metric, metric.replace("_val", "").replace("/", " / "))
        ax.set_title(metric_name_short, fontsize=14)
        ax.set_xlabel("Gap", fontsize=10)
        ax.set_ylabel("Accuracy", fontsize=10)
        ax.set_xticks(sorted(list(set(g for model in all_files for g in all_files[model]['files_by_gap']))))
        ax.set_ylim(0, 1.0)
        ax.grid(True, which="both", linestyle="--", linewidth=0.5)
        ax.legend()

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    output_filename = os.path.join(output_dir, "minigpt4_perf_vs_gap_all_metrics.png")
    plt.savefig(output_filename)
    plt.close()
    print(f"Generated combined plot: {output_filename}")

    # --- Save results to CSV ---
    for model_name, summary_data in all_summary_data.items():
        if summary_data:
            df = pd.DataFrame(summary_data)
            csv_filename = os.path.join(output_dir, f"minigpt4_{model_name}_performance_summary.csv")
            df.to_csv(csv_filename, index=False, float_format="%.4f")
            print(f"Saved summary statistics to: {csv_filename}")

if __name__ == "__main__":
    analyze_and_plot()