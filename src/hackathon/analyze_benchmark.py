import json

import matplotlib.pyplot as plt
import pandas as pd

from hackathon.benchmark import BenchmarkResult
from hackathon.hotpot_evalaute_f1 import f1_score

with open("benchmark_result.json", "r", encoding="utf-8") as f:
    benchmark_result_json = json.load(f)


benchmark_result = BenchmarkResult.model_validate(
    benchmark_result_json,
    context={"skip_abstract_event_validation": True},
)

scores = {}
for single_run in benchmark_result.runs:
    prediction = single_run.answer_summary
    ground_truth = single_run.dataset_row.answer
    likelihood = single_run.likelihood
    f1, precision, recall = f1_score(prediction, ground_truth)
    scores[single_run.dataset_row.id] = {
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "likelihood": likelihood,
    }

df_scores = pd.DataFrame(scores)

mean_scores = df_scores.mean()
print(mean_scores)


plot_df = df_scores.T
plot_df["likelihood_bin"] = pd.qcut(plot_df["likelihood"], q=10, duplicates="drop")
calibration_df = plot_df.groupby("likelihood_bin").agg(
    mean_likelihood=("likelihood", "mean"),
    mean_f1=("f1", "mean"),
)

plt.figure()
plt.plot(
    calibration_df["mean_likelihood"],
    calibration_df["mean_f1"],
    marker="o",
)
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("Mean likelihood")
plt.ylabel("Mean F1")
plt.title("Calibration plot: F1 vs likelihood")
plt.tight_layout()
plt.show()


brier_score = ((plot_df["likelihood"] - plot_df["f1"]) ** 2).mean()
print(f"Brier score: {brier_score}")

# breakpoint()
