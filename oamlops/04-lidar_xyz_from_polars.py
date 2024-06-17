# Databricks notebook source
# MAGIC %pip install --upgrade databricks-sdk
# MAGIC dbutils.library.restartPython()

# COMMAND ----------
# set the number of cores per task (default 1)
# spark.conf.set("spark.task.cpus", 2)

# COMMAND ----------

# MAGIC %pip install --upgrade databricks-sdk

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

from pyspark.sql import functions as F
import pyspark.sql.types as T

from utils import databricks_utils
from utils.databricks_utils import write_metadata
import os
import comlink
import asyncio
import equalaser
from equalaser._calibration import AzureLidarCalibrationDownloader, LidarCalibrationFetcherWithExternal
import nest_asyncio
import acircuit

nest_asyncio.apply()

import pandas as pd
import numpy as np
from typing import Tuple
from datetime import datetime

logging = databricks_utils.setup_logging(
    os.path.basename(dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get())
)

# COMMAND ----------

dbutils.widgets.text("session_name", "EL6W026_2023_11_06__11_46_37")
dbutils.widgets.text("inference.mlflow_model_run_id", "c4ddf6de80da4864bd85687aec176e26")
dbutils.widgets.text("run_id", "")
dbutils.widgets.text("databricks_job_run_id", "")
dbutils.widgets.text("databricks_job_parent_run_id", "")


dbutils.widgets.text("realignment.version", "0")
dbutils.widgets.text("realignment.key", "trajectory")
dbutils.widgets.text("realignment.type", "s2s")
dbutils.widgets.text("realignment.tsUrl", "http://service.trajectory-store-prod.tthad.net/trajectory-store")

bucket_size = 1
TIME_DELTA = 1  # segs

# COMMAND ----------




#get environment
dbutils.widgets.text("databricks_env", "") 
databricks_env =  dbutils.widgets.get("databricks_env") 

from config.settings import ConfigLoader
config_loader = ConfigLoader(databricks_env)
config = config_loader.get_config()
LIDAR_POLARS_TABLE_PATH = config['LIDAR_POLARS_TABLE_PATH']
LIDAR_XYZ_TABLE_PATH = config['LIDAR_XYZ_TABLE_PATH']
INFERENCE_TABLE_PATH = config['INFERENCE_TABLE_PATH']
RUN_METADATA_TABLE_PATH = config['RUN_METADATA_TABLE_PATH']
AZURE_LIDAR_CALIBRATION_SAS = config['AZURE_LIDAR_CALIBRATION_SAS']
CARPOSITIONS_TABLE_PATH = config['CARPOSITIONS_TABLE_PATH']
#TRAJECTORY_TABLE_PATH = config['TRAJECTORY_TABLE_PATH']
LASER_NAME = config['LASER_NAME']
SQLITE_FOLDER_PATH = config['SQLITE_FOLDER_PATH']
MOUNT_DESTINATION = config['MOUNT_DESTINATION']

session_name = dbutils.widgets.get("session_name")
mlflow_model_run_id = dbutils.widgets.get("inference.mlflow_model_run_id")
run_id = dbutils.widgets.get("run_id")
databricks_job_run_id = dbutils.widgets.get("databricks_job_run_id")
databricks_job_parent_run_id = dbutils.widgets.get("databricks_job_parent_run_id")

realignment_version = dbutils.widgets.get("realignment.version")
realignment_key = dbutils.widgets.get("realignment.key")
realignment_type = dbutils.widgets.get("realignment.type")
realignment_tsUrl = dbutils.widgets.get("realignment.tsUrl")

realignment = {
    "version": realignment_version,
    "key": realignment_key,
    "type": realignment_type,
    "tsUrl": realignment_tsUrl,
}

inference = {"mlflow_model_run_id": mlflow_model_run_id}

input_parameters = {
    "session_name": session_name,
    "inference.mlflow_model_run_id": mlflow_model_run_id,
    "run_id": run_id,
    "databricks_job_run_id": databricks_job_run_id,
    "run_metadata_table_path": RUN_METADATA_TABLE_PATH,
    "realignment.version": realignment_version,
    "realignment.key": realignment_key,
    "realignment.type": realignment_type,
    "realignment.tsUrl": realignment_tsUrl,
    "BUCKET_SIZE": bucket_size,
    "DEST_TABLE": LIDAR_XYZ_TABLE_PATH,
    "INFERENCE_TABLE": INFERENCE_TABLE_PATH,
    "CARPOSITIONS_TABLE": CARPOSITIONS_TABLE_PATH,
    "TIME_DELTA": TIME_DELTA,
}
cluster_id = dbutils.entry_point.getDbutils().notebook().getContext().tags().get("clusterId").get()

# COMMAND ----------

tenant_id = dbutils.secrets.get(scope="repository-secrets", key="reksio-app-tenant-id")
principal_id = dbutils.secrets.get(scope="repository-secrets", key="reksio-app-principal-id")
principal_secret = dbutils.secrets.get(scope="repository-secrets", key="reksio-app-principal-secret")


storage_acc = LIDAR_POLARS_TABLE_PATH.split("@")[1].split(".")[0]
# Access to the tables with SP
databricks_utils.set_storage_account_config(spark, storage_acc, principal_id, tenant_id, principal_secret)

# COMMAND ----------

from datetime import datetime

write_metadata(
    spark,
    run_metadata_table_path=RUN_METADATA_TABLE_PATH,
    run_id=run_id,
    session_name=session_name,
    workflow="lidar_xyz",
    event="started",
    event_time=datetime.now(),
    databricks_job_run_id=databricks_job_run_id,
    databricks_job_parent_run_id=databricks_job_parent_run_id,
    input_parameters=input_parameters,
    cluster_id=cluster_id,
)

# COMMAND ----------

lidar_polars = spark.read.format("delta").load(LIDAR_POLARS_TABLE_PATH).where(F.col("session_name") == session_name)

# COMMAND ----------

# MAGIC %md
# MAGIC # Get cameraframes

# COMMAND ----------

carpositions = spark.read.format("delta").load(CARPOSITIONS_TABLE_PATH).where(F.col("session_name") == session_name)
carpositions = (
    carpositions.withColumn("time_only", F.date_format("frametime", "HH:mm:ss.SSS")).withColumn(
        "frametime",
        F.split(F.col("time_only"), ":")[0].cast("int") * 3600  # hours to seconds
        + F.split(F.col("time_only"), ":")[1].cast("int") * 60  # minutes to seconds
        + F.split(F.split(F.col("time_only"), ":")[2], "\.")[0].cast("int")  # seconds
        + F.split(F.split(F.col("time_only"), ":")[2], "\.")[1].cast("double") / 1000,
    )
).select("session_name", "frame_index", "frametime")


inference_frames = spark.read.format("delta").load(INFERENCE_TABLE_PATH)
inference_frames = (
    inference_frames.filter(F.col("session_name") == session_name)
    .filter(F.col("mlflow_model_run_id") == mlflow_model_run_id)
    .select(["session_name", "frame_index"])
    .drop_duplicates()
)

carpositions = carpositions.join(inference_frames, how="inner", on=["session_name", "frame_index"])

# Read lidar table anti join inference to know frames that we need to generate
try:
    lidar_data = (
        spark.read.format("delta")
        .load(LIDAR_XYZ_TABLE_PATH)
        .select("session_name", "frame_index", "realignment_version")
    )
    lidar_data = lidar_data.where(F.col("realignment_version") == realignment_version)
    carpositions = carpositions.join(lidar_data, on=["session_name", "frame_index"], how="leftanti")

except Exception as ex:
    logging.info(f"Writing all lidars, ex: {ex}")

total_frames = carpositions.count()
logging.info(f"Processing lidar for total of {total_frames} inference frames")


# COMMAND ----------

# MAGIC %md
# MAGIC # Get polars and transform into xyz

# COMMAND ----------

session = comlink.SessionName(session_name)
track_source = comlink.TrackSource(
    version=int(realignment_version), key=realignment_key, type_=realignment_type, url=realignment_tsUrl
)
calibration_fetcher = LidarCalibrationFetcherWithExternal(
    ext_calibration_downloader=AzureLidarCalibrationDownloader(
        AZURE_LIDAR_CALIBRATION_SAS,
    )
)

sess_info = asyncio.run(
    equalaser._utils.SessInfo.fetch(session, calibration_fetcher=calibration_fetcher, track_source=track_source)
)

# COMMAND ----------

lidar_polars_with_buckets = lidar_polars.withColumn("time_bucket", (F.col("time") / bucket_size).cast("integer"))
carpositions_with_buckets = carpositions.withColumn(
    "frametime_bucket", (F.col("frametime") / bucket_size).cast("integer")
)
broadcast_carpositions_with_buckets = F.broadcast(
    carpositions_with_buckets.select("frame_index", "frametime", "frametime_bucket")
)

bucketed_join = lidar_polars_with_buckets.join(
    broadcast_carpositions_with_buckets,
    (
        (F.col("time_bucket") == F.col("frametime_bucket"))
        | (F.col("time_bucket") == F.col("frametime_bucket") - 1)
        | (F.col("time_bucket") == F.col("frametime_bucket") + 1)
    ),
    how="inner",
)

lidar_frames = bucketed_join.filter(
    (F.col("time") > F.col("frametime") - TIME_DELTA) & (F.col("time") < F.col("frametime") + TIME_DELTA)
)


# COMMAND ----------


def extract_xyz(lidar_frames):
    dtype = np.dtype([("distance", "<u2", (32,)), ("refl", "u1", (32,)), ("theta", "<u2"), ("time", "<f8")])
    selected_scans = np.array(
        list(zip(lidar_frames["distance"], lidar_frames["refl"], lidar_frames["theta"], lidar_frames["time"])),
        dtype=dtype,
    )
    world_xyz, _ = equalaser._utils.sensor_to_car_to_world(selected_scans, sess_info)
    return pd.DataFrame(
        {
            "session_name": lidar_frames["session_name"].iloc[0],
            "frame_index": lidar_frames["frame_index"].iloc[0],
            "frametime": lidar_frames["frametime"].iloc[0],
            "world_xyz": [world_xyz.tobytes()],
            "scan_time_in_seconds": [np.array(lidar_frames["time"]).tobytes()],
        }
    )


# COMMAND ----------

lidar_xyz = lidar_frames.groupby("session_name", "frame_index", "frametime").applyInPandas(
    extract_xyz,
    schema="session_name string, frame_index integer, frametime double, world_xyz binary,scan_time_in_seconds binary",
)
lidar_xyz = lidar_xyz.withColumn("realignment_version", F.lit(realignment_version))
lidar_xyz.write.format("delta").partitionBy("session_name", "realignment_version").mode("append").save(
    LIDAR_XYZ_TABLE_PATH
)

# COMMAND ----------

write_metadata(
    spark,
    run_metadata_table_path=RUN_METADATA_TABLE_PATH,
    run_id=run_id,
    session_name=session_name,
    workflow="lidar_xyz",
    event="done",
    event_time=datetime.now(),
    databricks_job_run_id=databricks_job_run_id,
    databricks_job_parent_run_id=databricks_job_parent_run_id,
    input_parameters=input_parameters,
    lidar={"lidar_frames_extracted": total_frames},
    realignment=realignment,
    inference=inference,
    cluster_id=cluster_id,
)
