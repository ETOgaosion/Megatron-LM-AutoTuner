from typing import List


def average_microbatch_metric(micro_batch_results: List[dict], key: str):
    assert len(micro_batch_results) > 0, "No micro batch results to average."
    total = 0.0
    for result in micro_batch_results:
        total += result[key]
    return total / len(micro_batch_results)
