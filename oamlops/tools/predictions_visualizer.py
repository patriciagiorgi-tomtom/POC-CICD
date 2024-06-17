# Databricks notebook source
# MAGIC %pip install geojson

# COMMAND ----------

from pyspark.sql.types import StringType
from pyspark.sql.functions import udf
import numpy as np
from PIL import Image, ImageDraw
import io
import os
import pyspark.sql.functions as F

# COMMAND ----------

import mlflow

mlflow_experiment_name = "/Users/chirag.garg@tomtom.com/road-furniture-mmdetection"
mlflow_run_name = "Plot predictions"

if mlflow.get_experiment_by_name(mlflow_experiment_name) is None:
    mlflow.create_experiment(mlflow_experiment_name)
mlflow.set_experiment(mlflow_experiment_name)
mlflow.start_run(run_name=mlflow_run_name)
mlflow_run_id = mlflow.active_run().info.run_id
mlflow_experiment_id = mlflow.active_run().info.experiment_id

# COMMAND ----------

mlflow_run_id

# COMMAND ----------

# MAGIC %md
# MAGIC ## Library for plotting

# COMMAND ----------

# MAGIC %md
# MAGIC ## Read input images and predictions

# COMMAND ----------

from road_furniture.configs.inference.storage_account_config import StorageAuthCredentials
from road_furniture.utils.general_utils import set_azure_storage_auth

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

session_name = "EL6W026_2023_11_06__11_21_17"
moma_inference_df = (
    spark.read.format("delta")
    .load("abfss://oamlops-bronze-dev@reksiodbxstorageaccount.dfs.core.windows.net/cameraframes.delta")
    .filter(F.col("session_name") == session_name)
)
predictions_df = (
    spark.read.format("delta")
    .load("abfss://oamlops-bronze-dev@reksiodbxstorageaccount.dfs.core.windows.net/inferences.delta")
    .filter(F.col("session_name") == session_name)
)
run_metadata_df = (
    spark.read.format("delta")
    .load("abfss://oamlops-bronze-dev@reksiodbxstorageaccount.dfs.core.windows.net/run_metadata.delta")
    .filter(F.col("session_name") == session_name)
)

# COMMAND ----------

output_dir_path = os.path.join("/dbfs/tmp/inference_ingolstad/predictions", f"viz_{session_name}")
os.makedirs(output_dir_path, exist_ok=True)

# COMMAND ----------


def get_bbox_coordinates(x_center, y_center, w, h):
    xmin = x_center - w / 2
    ymin = y_center - h / 2
    xmax = x_center + w / 2
    ymax = y_center + h / 2
    return xmin, ymin, xmax, ymax


# Define a UDF to process the image
def process_and_save_image(image_bytes, detections, frame_index):
    image = Image.open(io.BytesIO(image_bytes))
    draw = ImageDraw.Draw(image)
    for detection in detections:
        bbox = get_bbox_coordinates(detection["bbox_x"], detection["bbox_y"], detection["bbox_w"], detection["bbox_h"])
        class_name = detection["class"]
        score = np.round(detection["score"] * 100, 2)
        draw.rectangle([(bbox[0], bbox[1]), (bbox[2], bbox[3])], outline="blue", width=2)
        draw.text((bbox[0], bbox[1] - 10), f"id_{class_name}:pb_{score}", fill="green")

    # Save image locally
    img_save_path = os.path.join(output_dir_path, f"{frame_index}.png")
    image.save(img_save_path)

    return img_save_path


# Register the UDF
process_and_save_image_udf = udf(process_and_save_image, StringType())

# COMMAND ----------

predictions_df.select("frame_index", "session_name", "score").groupBy("frame_index").count().display()

# COMMAND ----------

viz_pred_df = predictions_df

# COMMAND ----------

prediction_images_df = viz_pred_df.groupBy("frame_index").agg(
    F.collect_list(F.struct(viz_pred_df.columns)).alias("detections")
)
prediction_images_df = prediction_images_df.join(moma_inference_df, "frame_index")
prediction_images_df = prediction_images_df.withColumn(
    "image_path", process_and_save_image_udf("panorama_img", "detections", "frame_index")
)
prediction_images_df.select("frame_index", "session_name", "image_path").display()
mlflow.log_artifact(output_dir_path)
