import json

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
    f1, precision, recall = f1_score(prediction, ground_truth)
    scores[single_run.dataset_row.id] = {
        "f1": f1,
        "precision": precision,
        "recall": recall,
    }

df_scores = pd.DataFrame(scores)

mean_scores = df_scores.mean()
print(mean_scores)
