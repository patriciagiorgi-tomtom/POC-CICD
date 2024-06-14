# Databricks notebook source
# MAGIC %pip install -U openmim
# MAGIC %pip install --upgrade mlflow

# COMMAND ----------

# MAGIC %sh
# MAGIC mim install mmdet

# COMMAND ----------

# MAGIC %pip install pynvml

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import mlflow
mlflow.enable_system_metrics_logging()

# COMMAND ----------

import os
notebook_dir_path = f"/Workspace/{os.path.dirname(dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get())}"
os.chdir(notebook_dir_path)

# COMMAND ----------

from oamlops.model_manager.model_registry import (
    ModelRegistryParameters,
    ModelRegistry,
)
from mmdet_visualization import MLflowVisBackendDatabricks
import mlflow

# COMMAND ----------

# MAGIC %md
# MAGIC # Widgets
# MAGIC - Use **full path of model configs** from a clone of latest main branch of https://github.com/tomtom-internal/uhd_map at _"/Workspace/Repos/uhd_map"_
# MAGIC - Provide the image width and height to be used for **inference inside the model** provided in model configs in **test_pipeline**

# COMMAND ----------

# Parameters for notebook
dbutils.widgets.text("inferencer_type", "DetInferencer")
dbutils.widgets.text("checkpoint_mlflow_run_id", "c4ddf6de80da4864bd85687aec176e26")
dbutils.widgets.text("model_temp_dir_path", "/dbfs/FileStore/model")
dbutils.widgets.text(
    "model_configs_path",
    "/Workspace/Repos/uhd_map/uhd_map/road_furniture/configs/cascade_rcnn/cascade-rcnn_r50_fpn_1x_zenseact_mapillary_binary_traffic_sign.py",
)
dbutils.widgets.text("model_name", "DetInferencer_test")
dbutils.widgets.text("device_type", "gpu")
dbutils.widgets.text("inference_image_width", '2666')
dbutils.widgets.text("inference_image_height", '1333')

# COMMAND ----------

args_dict = dbutils.notebook.entry_point.getCurrentBindings()
print(args_dict)

# COMMAND ----------

mlflow_experiment_name = "/Users/chirag.garg@tomtom.com/road-furniture-mmdetection"
mlflow_run_name = "Register Model"

if mlflow.get_experiment_by_name(mlflow_experiment_name) is None:
    mlflow.create_experiment(mlflow_experiment_name)
mlflow.set_experiment(mlflow_experiment_name)
mlflow.start_run(run_name=mlflow_run_name)
mlflow_run_id = mlflow.active_run().info.run_id
mlflow_experiment_id = mlflow.active_run().info.experiment_id

# COMMAND ----------

# MAGIC %md
# MAGIC # Register Model

# COMMAND ----------

registry_parameters = ModelRegistryParameters(**args_dict)

# COMMAND ----------

model_registry = ModelRegistry(registry_parameters)

# COMMAND ----------

model_info = model_registry.register_model(download_artifact=False)

# COMMAND ----------

print(f"Model info for inference {model_info}")

# COMMAND ----------

print(f"Metadata for inference {model_info.metadata}")

# COMMAND ----------

# MAGIC %md
# MAGIC # Load inferencer model

# COMMAND ----------

import mlflow

# COMMAND ----------

# model = mlflow.pyfunc.load_model(f"models:/DetInferencer_test/latest")
model = mlflow.pyfunc.load_model(model_uri=model_info.model_uri)

# COMMAND ----------

inferencer_test = model.unwrap_python_model().inferencer

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC # Test Model

# COMMAND ----------

from oamlops.utils.databricks_utils import set_storage_account_config, write_metadata
import pyspark.sql.functions as F

STORAGE_ACCOUNT = 'reksiodbxstorageaccount'

tenant_id = dbutils.secrets.get(scope="repository-secrets", key="reksio-app-tenant-id")
principal_id = dbutils.secrets.get(scope="repository-secrets", key="reksio-app-principal-id")
principal_secret = dbutils.secrets.get(scope="repository-secrets", key="reksio-app-principal-secret")

set_storage_account_config(spark, STORAGE_ACCOUNT, principal_id, tenant_id, principal_secret)

# COMMAND ----------

selected_sessions = ["EL7S238_2021_08_10__07_14_45"]
moma_inference_df = (
    spark.read.format("delta")
    .load("abfss://oamlops-bronze-dev@reksiodbxstorageaccount.dfs.core.windows.net/cameraframes.delta")
    .filter(F.col("session_name").isin(selected_sessions))
).limit(50)

# COMMAND ----------

from PIL import Image
import numpy
import io

image_bytes = moma_inference_df.select("panorama_img").rdd.flatMap(lambda x: x).collect()
images = [numpy.asarray(Image.open(io.BytesIO(byte)).resize((registry_parameters.inference_image_width, 
                                                            registry_parameters.inference_image_height))) for byte in image_bytes]

# COMMAND ----------

detections = inferencer_test(images, batch_size=4)

# COMMAND ----------

detections

# COMMAND ----------

mlflow.end_run()
