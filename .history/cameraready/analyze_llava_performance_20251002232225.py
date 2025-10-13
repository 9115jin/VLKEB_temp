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
            "dir": "results/results_sequencial/composition/lora",
            "files_by_gap": {
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
        },
        "ft": {
            "dir": "results/results_sequencial/composition/ft",
            "files_by_gap": {
                0: [
                    "250930_201451_ft_llava_port1_seqgap0_testnum500.json",
                    "251001_144854_ft_llava_port1_seqgap0_testnum500.json",
                    "251001_144913_ft_llava_port1_seqgap0_testnum500.json",
                    "251001_144926_ft_llava_port1_seqgap0_testnum500.json",
                    "251001_144943_ft_llava_port1_seqgap0_testnum500.json"
                ],
                10: [
                    "250930_215806_ft_llava_port1_seqgap10_testnum500.json",
                    "251001_163215_ft_llava_port1_seqgap10_testnum500.json",
                    "251001_163237_ft_llava_port1_seqgap10_testnum500.json",
                    "251001_163255_ft_llava_port1_seqgap10_testnum500.json",
                    "251001_163307_ft_llava_port1_seqgap10_testnum500.json"
                ],
                20: [
                    "250930_234206_ft_llava_port1_seqgap20_testnum500.json",
                    "251001_181614_ft_llava_port1_seqgap20_testnum500.json",
                    "251001_181723_ft_llava_port1_seqgap20_testnum500.json",
                    "251001_181725_ft_llava_port1_seqgap20_testnum500.json",
                    "251001_181728_ft_llava_port1_seqgap20_testnum500.json"
                ],
                50: [
                    "251001_012709_ft_llava_port1_seqgap50_testnum500.json",
                    "251001_200211_ft_llava_port1_seqgap50_testnum500.json",
                    "251001_200348_ft_llava_port1_seqgap50_testnum500.json",
                    "251001_200350_ft_llava_port1_seqgap50_testnum500.json",
                    "251001_200402_ft_llava_port1_seqgap50_testnum500.json"
                ],
                100: [
                    "251001_031600_ft_llava_port1_seqgap100_testnum500.json",
                    "251001_215146_ft_llava_port1_seqgap100_testnum500.json",
                    "251001_215341_ft_llava_port1_seqgap100_testnum500.json",
                    "251001_215346_ft_llava_port1_seqgap100_testnum500.json",
                    "251001_215426_ft_llava_port1_seqgap100_testnum500.json"
                ]
            }
        },
        "mend": {
            "dir": "results/results_sequencial/composition",
            "files_by_gap": {
                0: [
                    "test500_compositional250510_131529_MEND_llava_port1_seqgap0.json",
                    "test500_compositional251002_171200_MEND_llava_port1_seqgap0.json",
                    "test500_compositional251002_171319_MEND_llava_port1_seqgap0.json",
                    "test500_compositional251002_171329_MEND_llava_port1_seqgap0.json",
                    "test500_compositional251002_171340_MEND_llava_port1_seqgap0.json"
                ],
                10: [
                    "test500_compositional250510_142231_MEND_llava_port1_seqgap10.json",
                    "test500_compositional251002_174212_MEND_llava_port1_seqgap10.json",
                    "test500_compositional251002_174303_MEND_llava_port1_seqgap10.json",
                    "test500_compositional251002_174322_MEND_llava_port1_seqgap10.json",
                    "test500_compositional251002_174519_MEND_llava_port1_seqgap10.json"
                ],
                20: [
                    "test500_compositional250510_153239_MEND_llava_port1_seqgap20.json",
                    "test500_compositional251002_181131_MEND_llava_port1_seqgap20.json",
                    "test500_compositional251002_181353_MEND_llava_port1_seqgap20.json",
                    "test500_compositional251002_181357_MEND_llava_port1_seqgap20.json",
                    "test500_compositional251002_181643_MEND_llava_port1_seqgap20.json"
                ],
                50: [
                    "test500_compositional250510_163950_MEND_llava_port1_seqgap50.json",
                    "test500_compositional251002_184231_MEND_llava_port1_seqgap50.json",
                    "test500_compositional251002_184302_MEND_llava_port1_seqgap50.json",
                    "test500_compositional251002_184310_MEND_llava_port1_seqgap50.json",
                    "test500_compositional251002_184731_MEND_llava_port1_seqgap50.json"
                ],
                100: [
                    "test500_compositional250510_175320_MEND_llava_port1_seqgap100.json",
                    "test500_compositional251002_191205_MEND_llava_port1_seqgap100.json",
                    "test500_compositional251002_191441_MEND_llava_port1_seqgap100.json",
                    "test500_compositional251002_191546_MEND_llava_port1_seqgap100.json",
                    "test500_compositional251002_191727_MEND_llava_port1_seqgap100.json"
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
    fig.suptitle("Performance vs. Sequential Edit Gap (testnum=500)", fontsize=24, y=0.98)
    axes = axes.flatten()
    
    all_summary_data = defaultdict(list)
    model_colors = {
        "lora": "skyblue",
        "ft": "purple",
        "mend": "green"
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
    output_filename = os.path.join(output_dir, "perf_vs_gap_all_metrics.png")
    plt.savefig(output_filename)
    plt.close()
    print(f"Generated combined plot: {output_filename}")

    # --- Save results to CSV ---
    for model_name, summary_data in all_summary_data.items():
        if summary_data:
            df = pd.DataFrame(summary_data)
            csv_filename = os.path.join(output_dir, f"{model_name}_performance_summary.csv")
            df.to_csv(csv_filename, index=False, float_format="%.4f")
            print(f"Saved summary statistics to: {csv_filename}")

if __name__ == "__main__":
    analyze_and_plot()