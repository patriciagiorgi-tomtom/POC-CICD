# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC # Install MM Detection

# COMMAND ----------

# MAGIC %pip install -U openmim
# MAGIC %pip install --upgrade mlflow

# COMMAND ----------

# MAGIC %md
# MAGIC **For using MMDetection**

# COMMAND ----------

# MAGIC %sh
# MAGIC mim install mmengine
# MAGIC mim install "mmcv>=2.0.0"
# MAGIC mim install mmdet
# MAGIC mim install "mmpretrain>=1.0.0rc8"

# COMMAND ----------

# MAGIC %md
# MAGIC **For using utils**

# COMMAND ----------

# MAGIC %pip install geojson
# MAGIC %pip install pyproj
# MAGIC %pip install h3
# MAGIC %pip install haversine

# COMMAND ----------

# MAGIC %md
# MAGIC **For logging GPU Metrics**

# COMMAND ----------

# MAGIC %pip install pynvml

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import os

notebook_path = f"/Workspace/{os.path.dirname(dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get())}"
os.chdir(notebook_path)

# COMMAND ----------

import os
import shutil
import mlflow
import torch
import pandas
import time
from pyspark.ml.feature import StringIndexer
from petastorm.spark import SparkDatasetConverter, make_spark_converter
from road_furniture.utils.general_utils import set_azure_storage_auth
from oamlops.moma_inference.inference_parameters import (
    InferenceJobParameters,
    JobParametersProcessor,
)
from oamlops.moma_inference.moma_inferencer import execute_moma_inference

from road_furniture.configs.inference.storage_account_config import StorageAuthCredentials
from oamlops.moma_inference.detections_processor import create_prediction_df

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC # Init widgets

# COMMAND ----------

# Path to model_configs
dbutils.widgets.text("mlflow_experiment_name", "/Users/chirag.garg@tomtom.com/road-furniture-mmdetection")
dbutils.widgets.text(
    "model_configs", "../../configs/cascade_rcnn/cascade-rcnn_r50_fpn_1x_zenseact_mapillary_binary_traffic_sign.py"
)
# The main dir to save inference results
dbutils.widgets.text("inference_data_output_path", "")
# Session names to do the inference for them
dbutils.widgets.text("inference_data_input_path", "")
# temp folder to save the mlflow artifacts to load the model checkpoint
dbutils.widgets.text("model_temp_dir_path", "/dbfs/tmp/inference_poc/model_temp_dir")
# mlflow run id containing the checkpoint to do inference
dbutils.widgets.text("mlflow_run_id", "")
dbutils.widgets.text("confidence_threshold", "0.6")
dbutils.widgets.text("device", "gpu")
dbutils.widgets.dropdown(
    "ci_cd_pipeline",
    "False",
    ["True", "False"],
)
dbutils.widgets.dropdown(
    "inferencer",
    "DetInferencer",
    ["DetInferencer", "TensorDetInferencer"],
)
dbutils.widgets.text("mlflow_run_name", "2d_detection_inference_test")
dbutils.widgets.text("inference_batch_size", "100")
dbutils.widgets.text("model_batch_size", "10")
dbutils.widgets.text("petastorm_cache_dir_path", "file:///dbfs/tmp/inference_poc/petastorm/cache")

# COMMAND ----------

args_dict = dbutils.notebook.entry_point.getCurrentBindings()
print(args_dict)

# COMMAND ----------

# MAGIC %md
# MAGIC **Enable MLflow Logging**

# COMMAND ----------

mlflow.enable_system_metrics_logging()


# COMMAND ----------

# MAGIC %md
# MAGIC Load Job parameters and sanitize them

# COMMAND ----------

job_params = InferenceJobParameters(**args_dict)
parameters_sanitizer = JobParametersProcessor(job_params)
job_params.mlflow_run_name = (
    f"{job_params.mlflow_run_name}_p{job_params.inference_batch_size}_m{job_params.model_batch_size}"
)
checkpoint_mlflow_run_id = parameters_sanitizer.get_checkpoint_run_id()
ci_cd_pipeline = parameters_sanitizer.set_ci_cd_pipeline()

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC # Asserts

# COMMAND ----------

from oamlops.road_furniture.utils.general_utils import check_config_classes_orders

check_config_classes_orders(job_params.model_configs)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC # Start Experiment

# COMMAND ----------

# TODO: At some point we want to write our own wrapper around MLlfow to define this logic only once. For now we copy it
#  between notebooks until we have a more stable pipeline.

if ci_cd_pipeline:
    mlflow.start_run(run_id=checkpoint_mlflow_run_id)
    print(f"Started MLflow run with id: {checkpoint_mlflow_run_id}")
else:
    if mlflow.get_experiment_by_name(job_params.mlflow_experiment_name) is None:
        mlflow.create_experiment(job_params.mlflow_experiment_name)

    mlflow.set_experiment(job_params.mlflow_experiment_name)
    mlflow.start_run(run_name=job_params.mlflow_run_name)
    mlflow_run_id = mlflow.active_run().info.run_id

# COMMAND ----------

# MAGIC %md
# MAGIC ###Update job parameters for current run_id

# COMMAND ----------

job_params.petastorm_cache_dir_path = os.path.join(job_params.petastorm_cache_dir_path, mlflow_run_id)
job_params.model_temp_dir_path = os.path.join(job_params.model_temp_dir_path, mlflow_run_id)
checkpoint_artifact_dir_path = os.path.join(job_params.model_temp_dir_path, checkpoint_mlflow_run_id)

# COMMAND ----------

# MAGIC %md
# MAGIC # Prepare Input data

# COMMAND ----------

# MAGIC %md
# MAGIC ##Authenticate Spark to Storage

# COMMAND ----------

set_azure_storage_auth(
    spark=spark,
    dbutils=dbutils,
    storage_account=StorageAuthCredentials.storage_account_name,
    secrets_scope=StorageAuthCredentials.secrets_scope,
    tenant_id_name=StorageAuthCredentials.tenant_id,
    principal_id_name=StorageAuthCredentials.principal_id,
    principal_secret_name=StorageAuthCredentials.principal_secret,
)

# COMMAND ----------

# MAGIC %md
# MAGIC ##Load raw camera frames data

# COMMAND ----------

moma_inference_df = spark.read.format("delta").load(job_params.inference_data_input_path)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Index Sessions
# MAGIC - Use StringIndexer to convert the session_name string column to a index so that pytorch can handle numerical format later on

# COMMAND ----------

session_indexer = StringIndexer(inputCol="session_name", outputCol="session_name_index")
moma_inference_df = session_indexer.fit(moma_inference_df).transform(moma_inference_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ##Cache petastorm data and convert to torch format

# COMMAND ----------


spark.conf.set(SparkDatasetConverter.PARENT_CACHE_DIR_URL_CONF, job_params.petastorm_cache_dir_path)
converter_inference_data = make_spark_converter(moma_inference_df)

# COMMAND ----------

# MAGIC %md
# MAGIC #PetaStorm Inference

# COMMAND ----------

# MAGIC %md
# MAGIC ###Download model artifact

# COMMAND ----------

mlflow.artifacts.download_artifacts(
    run_id=checkpoint_mlflow_run_id, artifact_path="/", dst_path=checkpoint_artifact_dir_path
)

# COMMAND ----------

# MAGIC %md
# MAGIC ###Perform inference
# MAGIC - Load data and perform batch inference

# COMMAND ----------

mlflow.log_metric("inference_start", time.time())

# COMMAND ----------

inference_detections, inference_frame_indices, image_sizes, session_indices, final_execution_time = (
    execute_moma_inference(
        job_parameters=job_params,
        notebook_dir_path=notebook_path,
        checkpoint_artifact_dir_path=checkpoint_artifact_dir_path,
        converter_inference_data=converter_inference_data,
        number_of_gpu_workers=torch.cuda.device_count(),
    )
)

# COMMAND ----------

mlflow.log_metric("inference_end", time.time())
mlflow.log_table(data=pandas.DataFrame(final_execution_time), artifact_file="batch_inference_profile.json")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Delete Cache Data

# COMMAND ----------

converter_inference_data.delete()

# COMMAND ----------

# MAGIC %md
# MAGIC #Process Detections

# COMMAND ----------

# MAGIC %md
# MAGIC ##Convert predictions to spark dataframe

# COMMAND ----------

predictions_df = create_prediction_df(
    spark, inference_detections, inference_frame_indices, image_sizes, job_params, mlflow_run_id, session_indices
)
predictions_df = predictions_df.join(
    moma_inference_df.select("session_name", "session_name_index", "frame_index"),
    on=["session_name_index", "frame_index"],
).drop("session_name_index")

# COMMAND ----------

predictions_df.display()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Write dataframe to storage

# COMMAND ----------

predictions_df.write.format("delta").mode("overwrite").option("overwriteSchema", "true").save(
    job_params.inference_data_output_path
)

# COMMAND ----------

dbutils.jobs.taskValues.set(key="mlflow_run_id", value=mlflow_run_id)
dbutils.jobs.taskValues.set(key="inference_work_dir", value=job_params.inference_data_output_path)
dbutils.jobs.taskValues.set(key="moma_sessions_root", value=job_params.inference_data_input_path)

# COMMAND ----------

if os.path.exists(checkpoint_artifact_dir_path):
    shutil.rmtree(checkpoint_artifact_dir_path)

# COMMAND ----------

mlflow.end_run()
