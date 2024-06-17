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
# MAGIC mim install mmdet
# MAGIC mim install "mmpretrain>=1.0.0rc8"

# COMMAND ----------

# MAGIC %md
# MAGIC **For logging GPU Metrics**

# COMMAND ----------

# MAGIC %pip install pynvml

# COMMAND ----------

# MAGIC %md
# MAGIC For databricks-sdk

# COMMAND ----------

# MAGIC %pip install --upgrade databricks-sdk

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import os

notebook_path = f"/Workspace/{os.path.dirname(dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get())}"
os.chdir(notebook_path)

# COMMAND ----------

import os
import mlflow
import torch
import time
from petastorm.spark import SparkDatasetConverter, make_spark_converter
from oamlops.moma_inference.inference_parameters import (
    InferenceJobParameters,
)
from oamlops.moma_inference.moma_inferencer import execute_moma_inference

from oamlops.moma_inference.classification.classifications_processor import create_classification_df
from oamlops.utils.databricks_utils import write_metadata, set_storage_account_config
from oamlops.moma_inference.classification.bbox_cropper import process_bboxes
import pyspark.sql.functions as F

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC # Init widgets

# COMMAND ----------

# Path to model_configs
dbutils.widgets.text("session_name", "EL7S242_2021_06_13__09_25_25")
dbutils.widgets.text("run_id", "")
dbutils.widgets.text("inference.mlflow_model_run_id", "models:/classifier_test/2")
dbutils.widgets.text("panorama.zoom_level", "1")
dbutils.widgets.text("databricks_job_run_id", "")
dbutils.widgets.text("databricks_job_parent_run_id", "")

# COMMAND ----------

# get environment
dbutils.widgets.text("databricks_env", "dev")
databricks_env = dbutils.widgets.get("databricks_env")

from config.settings import ConfigLoader

config_loader = ConfigLoader(databricks_env)
config = config_loader.get_config()
STORAGE_ACCOUNT = config["STORAGE_ACCOUNT"]
PANORAMA_TABLE_PATH = config["PANORAMA_TABLE_PATH"]
INFERENCE_TABLE_PATH = config["INFERENCE_TABLE_PATH"]
RUN_METADATA_TABLE_PATH = config["RUN_METADATA_TABLE_PATH"]

# ZOOM_LEVEL_DICT = config['ZOOM_LEVEL_DICT']
ZOOM_LEVEL_DICT = {1: config["ZOOM_LEVEL_DICT"]}

INFERENCE_INPUT_IMG_SIZE = config["INFERENCE_INPUT_IMG_SIZE"]

INFERENCE_MODEL_IMG_SIZE = config["INFERENCE_MODEL_IMG_SIZE"]

CLASSIFICATION_TABLE_PATH = config["CLASSIFICATION_TABLE_PATH"]

args_dict = {
    "model_mlflow_run_id": dbutils.widgets.get("inference.mlflow_model_run_id"),
    "session_name": dbutils.widgets.get("session_name"),
    "run_id": dbutils.widgets.get("run_id"),
    "databricks_job_run_id": dbutils.widgets.get("databricks_job_run_id"),
    "databricks_job_parent_run_id": dbutils.widgets.get("databricks_job_parent_run_id"),
    "panorama_zoom_level": dbutils.widgets.get("panorama.zoom_level"),
    "mlflow_experiment_name": "/Users/chirag.garg@tomtom.com/road-furniture-classification",
    "inference_data_output_path": INFERENCE_TABLE_PATH,
    "inference_data_input_path": PANORAMA_TABLE_PATH,
    "classification_data_output_path": CLASSIFICATION_TABLE_PATH,
    "confidence_threshold": "0.6",
    "device": "gpu",
    "inferencer": "ImageClassificationInferencer",
    "mlflow_run_name": "2d_sign_classification",
    "panorama_zoom_size": ZOOM_LEVEL_DICT[
        int(dbutils.widgets.get("panorama.zoom_level"))
    ],
    "original_image_size": INFERENCE_INPUT_IMG_SIZE,
    "rescale_image_size": INFERENCE_MODEL_IMG_SIZE,
    "petastorm_cache_dir_path": "file:///dbfs/tmp/inference_poc/petastorm/cache",
    "run_metadata_table_path": RUN_METADATA_TABLE_PATH,
    "inference_batch_size" : 2000,
    "model_batch_size" : 2000,
    "mount_source_point": "abfss://oamlops-test@reksiodbxstorageaccount.dfs.core.windows.net/", 
    "mount_destination_point": "/mnt/moma_crops_test",
    "save_bbox_zone": "EURO", 
    "save_bbox" : True,
}

print(args_dict)

from datetime import datetime

start_time = datetime.now()

print(f"Inference for run {args_dict['run_id']} at {start_time}")

cluster_id = (
    dbutils.entry_point.getDbutils()
    .notebook()
    .getContext()
    .tags()
    .get("clusterId")
    .get()
)

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
job_params.mlflow_run_name = f"{job_params.mlflow_run_name}_p{job_params.inference_batch_size}_m{job_params.model_batch_size}"

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC # Start Experiment

# COMMAND ----------

if mlflow.get_experiment_by_name(job_params.mlflow_experiment_name) is None:
    mlflow.create_experiment(job_params.mlflow_experiment_name)

mlflow.set_experiment(job_params.mlflow_experiment_name)
mlflow.start_run(run_name=job_params.mlflow_run_name)
mlflow_run_id = mlflow.active_run().info.run_id
print(f"Started MLflow run with id: {mlflow_run_id}")
mlflow_experiment_id = mlflow.active_run().info.experiment_id

# COMMAND ----------

# MAGIC %md
# MAGIC ###Update job parameters for current run_id

# COMMAND ----------

job_params.petastorm_cache_dir_path = os.path.join(
    job_params.petastorm_cache_dir_path, mlflow_run_id
)

# COMMAND ----------

# MAGIC %md
# MAGIC # Prepare Input data

# COMMAND ----------

# MAGIC %md
# MAGIC ##Authenticate Spark to Storage

# COMMAND ----------

tenant_id = dbutils.secrets.get(scope="repository-secrets", key="reksio-app-tenant-id")
principal_id = dbutils.secrets.get(
    scope="repository-secrets", key="reksio-app-principal-id"
)
principal_secret = dbutils.secrets.get(
    scope="repository-secrets", key="reksio-app-principal-secret"
)

set_storage_account_config(
    spark, STORAGE_ACCOUNT, principal_id, tenant_id, principal_secret
)

if job_params.save_bbox:
    from oamlops.utils.databricks_utils import mount_fs
    mount_fs(spark, principal_id, tenant_id, principal_secret, job_params.mount_source_point, job_params.mount_destination_point)

# COMMAND ----------

write_metadata(
    spark,
    run_metadata_table_path=RUN_METADATA_TABLE_PATH,
    run_id=job_params.run_id,
    session_name=job_params.session_name,
    workflow="classification",
    event="started",
    event_time=datetime.now(),
    input_parameters=args_dict,
    databricks_job_run_id=job_params.databricks_job_run_id,
    databricks_job_parent_run_id=job_params.databricks_job_parent_run_id,
    cluster_id=cluster_id,
)

# COMMAND ----------

selected_sessions = [job_params.session_name]
predictions_df = (
    spark.read.format("delta")
    .load(INFERENCE_TABLE_PATH)
    .filter(F.col("session_name").isin(selected_sessions))
)
total_inferences = predictions_df.count()
print(f"{total_inferences=}")

# COMMAND ----------

try:
    already_predicted = (
        spark.read.format("delta")
        .load(job_params.classification_data_output_path)
        .filter(F.col("session_name") == job_params.session_name)
        .filter(F.col("mlflow_model_run_id") == job_params.model_mlflow_run_id)
        .filter(F.col("zoom_level") == job_params.panorama_zoom_level)
        .count()
    )
except:
    print("It looks like there is no classification table. We are good to go.")
    already_predicted = 0

if already_predicted > 0:
    print(
        "Looks like this version of the model has already been used with this session... skipping classification"
    )
    predictions_df = predictions_df.limit(0)

    mlflow.end_run()
    write_metadata(
        spark,
        run_metadata_table_path=job_params.run_metadata_table_path,
        run_id=job_params.run_id,
        session_name=job_params.session_name,
        workflow="classification",
        event="skipped",
        event_time=datetime.now(),
        input_parameters=args_dict,
        databricks_job_run_id=job_params.databricks_job_run_id,
        databricks_job_parent_run_id=job_params.databricks_job_parent_run_id,
        classification={
            "mlflow_classification_run_id": mlflow_run_id,
            "mlflow_classification_experiment_id": mlflow_experiment_id,
            "mlflow_model_run_id": job_params.model_mlflow_run_id,
            "total_frames_with_inference": 0,
            "total_inferences": 0,
        },
        panorama={"zoom_level": job_params.panorama_zoom_level},
        cluster_id=cluster_id,
    )
    dbutils.notebook.exit(0)


else:
    print(
        f"Running classification on {total_inferences} bboxes for session {selected_sessions}"
    )


# COMMAND ----------

# MAGIC %md
# MAGIC ##Crop BBoxes
# MAGIC

# COMMAND ----------

bbox_df, predictions_df = process_bboxes(spark=spark,job_parameters=job_params, predictions_df=predictions_df)
bbox_df.repartition(2).display()

# COMMAND ----------

# MAGIC %md
# MAGIC ##Cache petastorm data and convert to torch format

# COMMAND ----------

spark.conf.set(
    SparkDatasetConverter.PARENT_CACHE_DIR_URL_CONF, job_params.petastorm_cache_dir_path
)
converter_inference_data = make_spark_converter(bbox_df)

# COMMAND ----------

# MAGIC %md
# MAGIC #PetaStorm Inference

# COMMAND ----------

mlflow.log_metric("classification_start", time.time())

# COMMAND ----------

# MAGIC %md
# MAGIC ###Load model from mlflow

# COMMAND ----------

model = mlflow.pyfunc.load_model(job_params.model_mlflow_run_id)

# COMMAND ----------

# MAGIC %md
# MAGIC ###Perform inference
# MAGIC

# COMMAND ----------

inference_detections, inference_frame_indices = execute_moma_inference(
    job_parameters=job_params,
    converter_inference_data=converter_inference_data,
    model=model,
    number_of_gpu_workers=torch.cuda.device_count(),
    index_column="bbox_index",
    input_image_column="cropped_image",
    output_image_column="image",
)

# COMMAND ----------

mlflow.log_metric("classification_end", time.time())

# COMMAND ----------

# MAGIC %md
# MAGIC ### Delete Cache Data

# COMMAND ----------

try:
    converter_inference_data.delete()
except:
    print("Failed deleting cache")

# COMMAND ----------

# MAGIC %md
# MAGIC #Process Detections

# COMMAND ----------

# MAGIC %md
# MAGIC ##Convert predictions to spark dataframe

# COMMAND ----------

classification_df = (
    create_classification_df(
        spark, inference_detections, inference_frame_indices, model.metadata.metadata['classes']
    )
    .join(predictions_df, on="bbox_index")
    .withColumnRenamed("mlflow_inference_run_id", "mlflow_detection_run_id")
    .withColumnRenamed("mlflow_inference_experiment_id", "mlflow_detection_experiment_id")
    .withColumnRenamed("mlflow_model_run_id", "detection_model_uri")
    .withColumnRenamed("class", "detection_class")
    .withColumnRenamed("score", "detection_score")
    .withColumnRenamed("class_label", "classification_label")
    .withColumnRenamed("class_score", "classification_score")
    .withColumn("classifier_model_uri", F.lit(job_params.model_mlflow_run_id))
    .withColumn("run_id", F.lit(job_params.run_id))
    .withColumn("mlflow_classification_run_id", F.lit(mlflow_run_id))
    .withColumn("mlflow_classification_experiment_id", F.lit(mlflow_experiment_id))
    .drop('bbox_temp_path')
    .repartition(1)
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Write dataframe to storage

# COMMAND ----------

columns_order = [
    "session_name",
    "run_id",
    "mlflow_detection_run_id",
    "mlflow_detection_experiment_id",
    "detection_model_uri",
    "mlflow_classification_run_id",
    "mlflow_classification_experiment_id",
    "classifier_model_uri",
    "zoom_level",
    "frame_index",
    "bbox_index",
    "detection_score",
    "detection_class",
    "bbox_x",
    "bbox_y",
    "bbox_w",
    "bbox_h",
    "classification_label",
    "classification_score",
    "bbox_crop_path"
]
partitions = ["session_name"]
classification_df = classification_df.select(*columns_order).orderBy('frame_index', 'bbox_index')

# COMMAND ----------

classification_df.write.format("delta").mode(
    "append"
).partitionBy(*partitions).save(job_params.classification_data_output_path)
frames_with_inference = classification_df.select("frame_index").distinct().count()
total_inferences = classification_df.count()

# COMMAND ----------

write_metadata(
    spark,
    run_metadata_table_path=RUN_METADATA_TABLE_PATH,
    run_id=job_params.run_id,
    session_name=job_params.session_name,
    workflow="inference",
    event="done",
    event_time=datetime.now(),
    input_parameters=args_dict,
    databricks_job_run_id=job_params.databricks_job_run_id,
    databricks_job_parent_run_id=job_params.databricks_job_parent_run_id,
    classification={
        "mlflow_classification_run_id": mlflow_run_id,
        "mlflow_classification_experiment_id": mlflow_experiment_id,
        "mlflow_model_run_id": job_params.model_mlflow_run_id,
        "total_frames_with_inference": frames_with_inference,
        "total_inferences": total_inferences,
    },
    panorama={"zoom_level": job_params.panorama_zoom_level},
    cluster_id=cluster_id,
)

# COMMAND ----------

mlflow.end_run()
