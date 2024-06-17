import mlflow
import pandas as pd
import pyspark.sql.functions as F


def get_metric_history_df(
        run_id: str, metrics: list | None = None, gpu_count: int = 1
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
                    "mlflow_inference_run_name": run_name,
                    "mlflow_inference_run_id": run_id,
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

    df_avg = df_filtered.groupby(["mlflow_inference_run_name", "mlflow_inference_run_id", "metric_key"])[
        "value"].mean().reset_index()
    return metric_history_df, run.data, df_avg


def get_hardware_moniotring_metrics(spark, run_metadata_path, run_id):
    run_metadata = spark.read.format('delta').load(run_metadata_path).filter(F.col('run_id') == run_id) \
        .filter((F.col('event') == 'done') & (F.col('workflow') == 'inference'))

    # Explode inference clumn
    inference_metrics_df = run_metadata.select('*',
                                               F.col("inference.mlflow_inference_run_id").alias(
                                                   "mlflow_inference_run_id"),
                                               F.col("inference.mlflow_inference_experiment_id").alias(
                                                   "mlflow_inference_experiment_id"),
                                               F.col("inference.mlflow_model_run_id").alias("mlflow_model_run_id"),
                                               F.col("inference.total_frames").alias("total_frames"),
                                               F.col("inference.total_frames_with_inference").alias(
                                                   "total_frames_with_inference"),
                                               F.col("inference.total_inferences").alias("total_inferences")).drop(
        F.col('inference'))

    mlflow_run_id_df = inference_metrics_df.select('mlflow_inference_run_id').toPandas()

    df_avg_list = []
    for run_id in mlflow_run_id_df['mlflow_inference_run_id']:
        _, _, df_avg = get_metric_history_df(run_id)
        df_avg_list.append(df_avg)
    final_avg_df = pd.concat(df_avg_list)
    metrics_avg_df = spark.createDataFrame(final_avg_df)

    inference_metrics_df = inference_metrics_df.join(metrics_avg_df, on='mlflow_inference_run_id')
    return inference_metrics_df
