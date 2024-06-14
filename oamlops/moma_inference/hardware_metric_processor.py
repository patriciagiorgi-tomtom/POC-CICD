import mlflow
import pandas as pd


def get_metric_history_df(
    run_id: str, metrics: list | None, gpu_count: int = 1
) -> tuple[pd.DataFrame, dict, pd.DataFrame]:

    if metrics is None:
        metrics = []
        for gpu in range(gpu_count):
            metrics.extend([f"system/gpu_{gpu}_utilization_percentage", f"system/gpu_{gpu}_memory_usage_percentage"])
    metric_history = []
    run = mlflow.get_run(run_id)
    for metric_key in metrics:

        history = mlflow.MlflowClient().get_metric_history(run_id, metric_key)
        run_name = run.info.run_name
        for metric in history:
            metric_history.append(
                {
                    "run_name": run_name,
                    "run_id": run_id,
                    "metric_key": metric_key,
                    "timestamp": metric.timestamp,
                    "step": metric.step,
                    "value": metric.value,
                }
            )

    metric_history_df = pd.DataFrame(metric_history)
    start_time = run.data.metrics.get("inference_start")
    end_time = run.data.metrics.get("inference_end")

    df_filtered = metric_history_df[
        (metric_history_df["timestamp"] >= start_time * 1000) & (metric_history_df["timestamp"] <= end_time * 1000)
    ]

    df_avg = df_filtered.groupby(["run_name", "run_id", "metric_key"])["value"].mean().reset_index()
    return metric_history_df, run.data, df_avg
