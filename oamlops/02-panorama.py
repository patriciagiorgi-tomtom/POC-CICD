# Databricks notebook source

# COMMAND ----------

# MAGIC %pip install --upgrade databricks-sdk

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

#get environment
dbutils.widgets.text("databricks_env", "") 
databricks_env =  dbutils.widgets.get("databricks_env") 

from config.settings import ConfigLoader
config_loader = ConfigLoader(databricks_env)
config = config_loader.get_config()
STORAGE_ACCOUNT = config['STORAGE_ACCOUNT']
PANORAMA_TABLE_PATH = config['PANORAMA_TABLE_PATH']
SQLITE_FOLDER_PATH = config['SQLITE_FOLDER_PATH']
MOUNT_DESTINATION = config['MOUNT_DESTINATION']
RUN_METADATA_TABLE_PATH = config['RUN_METADATA_TABLE_PATH']
ZOOM_LEVEL_DICT = config['ZOOM_LEVEL_DICT']
INFERENCE_INPUT_IMG_SIZE =  config['INFERENCE_INPUT_IMG_SIZE']






# COMMAND ----------

dbutils.widgets.text("session_name", "EL2GM02_2023_07_28__20_01_50")
dbutils.widgets.text("panorama.zoom_level", "1")

dbutils.widgets.text("realignment.version", "")
dbutils.widgets.text("realignment.key", "")
dbutils.widgets.text("realignment.type", "")
dbutils.widgets.text("realignment.tsUrl", "")
dbutils.widgets.text("inference.mlflow_model_run_id", "")

dbutils.widgets.text("databricks_job_run_id", "")
dbutils.widgets.text("databricks_job_parent_run_id", "")

dbutils.widgets.text("run_id", "")

# COMMAND ----------

zoom_level = int(dbutils.widgets.get("panorama.zoom_level"))
session_name = dbutils.widgets.get("session_name")
databricks_job_run_id = dbutils.widgets.get("databricks_job_run_id")
databricks_job_parent_run_id = dbutils.widgets.get("databricks_job_parent_run_id")

run_id = dbutils.widgets.get("run_id")

realignment_version = dbutils.widgets.get("realignment.version")
realignment_key = dbutils.widgets.get("realignment.key")
realignment_type = dbutils.widgets.get("realignment.type")
realignment_tsUrl = dbutils.widgets.get("realignment.tsUrl")

inference_mlflow_model_run_id = dbutils.widgets.get("inference.mlflow_model_run_id")

BATCH_SIZE = 256

input_parameters = {
    "zoom_level": zoom_level,
    "session_name": session_name,
    "batch_size": BATCH_SIZE,
    "cameraframes_table_path": PANORAMA_TABLE_PATH,
    "run_metadata_table_path": RUN_METADATA_TABLE_PATH,
    "storage_account": STORAGE_ACCOUNT,
    "mount_source": SQLITE_FOLDER_PATH,
    "mount_destination": MOUNT_DESTINATION,
    "databricks_job_run_id": databricks_job_run_id,
    "realignment.version": realignment_version,
    "realignment.key": realignment_key,
    "realignment.type": realignment_type,
    "realignment.tsUrl": realignment_tsUrl,
    "inference.mlflow_model_run_id": inference_mlflow_model_run_id,
    "run_id": run_id,
}

cluster_id = dbutils.entry_point.getDbutils().notebook().getContext().tags().get("clusterId").get()

# COMMAND ----------

from datetime import datetime
import uuid

import pyspark.sql.functions as F

from process_panoramas.sqlite_to_delta import (
    retrieve_camera_frames,
    process_frames,
)
from utils.databricks_utils import set_storage_account_config, write_metadata, trigger_job

print(f"Starting run {run_id}")

# COMMAND ----------

# DBTITLE 1,set credentials and mount fs
tenant_id = dbutils.secrets.get(scope="repository-secrets", key="reksio-app-tenant-id")
principal_id = dbutils.secrets.get(scope="repository-secrets", key="reksio-app-principal-id")
principal_secret = dbutils.secrets.get(scope="repository-secrets", key="reksio-app-principal-secret")

set_storage_account_config(spark, STORAGE_ACCOUNT, principal_id, tenant_id, principal_secret)

configs = {
    "fs.azure.account.auth.type": "OAuth",
    "fs.azure.account.oauth.provider.type": "org.apache.hadoop.fs.azurebfs.oauth2.ClientCredsTokenProvider",
    "fs.azure.account.oauth2.client.id": principal_id,
    "fs.azure.account.oauth2.client.secret": principal_secret,
    "fs.azure.account.oauth2.client.endpoint": f"https://login.microsoftonline.com/{tenant_id}/oauth2/token",
}


try:
    dbutils.fs.mount(source=SQLITE_FOLDER_PATH, mount_point=MOUNT_DESTINATION, extra_configs=configs)
except Exception as e:
    print(e)
    if "Directory already mounted" not in e.__str__():
        raise RuntimeError("An Error occured while mounting the blob")

# COMMAND ----------

from utils.databricks_utils import get_session_absolute_path_from_name

session_absolute_path = "/dbfs" + get_session_absolute_path_from_name(session_name, MOUNT_DESTINATION) + ".SQLITE"

print(f"Starting process for session_name '{session_name}' from {session_absolute_path}")

write_metadata(
    spark,
    run_metadata_table_path=RUN_METADATA_TABLE_PATH,
    run_id=run_id,
    session_name=session_name,
    workflow="panorama",
    event="started",
    event_time=datetime.now(),
    databricks_job_run_id=databricks_job_run_id,
    databricks_job_parent_run_id=databricks_job_parent_run_id,
    input_parameters=input_parameters,
    cluster_id=cluster_id,
)

# # COMMAND ----------
# year = session_name.split("_")[1]
# date_str = "_".join(session_name.split("_")[1:4])

# # COMMAND ----------
# %ls /dbfs{MOUNT_DESTINATION}

# # COMMAND ----------
# %ls /dbfs{MOUNT_DESTINATION}/{year}

# # COMMAND ----------
# %ls /dbfs{MOUNT_DESTINATION}/{year}/{date_str}


# COMMAND ----------

# DBTITLE 1,cameraframes
print("Checking if session_name already exists in cameraframes")
try:
    prev_cameraframes = (
        spark.read.format("delta")
        .load(PANORAMA_TABLE_PATH)
        .select("session_name")
        .filter(F.col("session_name") == session_name)
        .count()
    )
except:
    print("Looks like cameraframes table does not exist, we are good to write")
    prev_cameraframes = 0

if prev_cameraframes > 1:
    print(f"Sorry :( session_name '{session_name}' already exists in cameraframes. I'm skipping this one...")
    total_frames = 0
    write_metadata(
        spark,
        run_metadata_table_path=RUN_METADATA_TABLE_PATH,
        run_id=run_id,
        session_name=session_name,
        workflow="panorama",
        event="skipped",
        event_time=datetime.now(),
        databricks_job_run_id=databricks_job_run_id,
        databricks_job_parent_run_id=databricks_job_parent_run_id,
        input_parameters=input_parameters,
        panorama={"zoom_level": zoom_level, "total_frames": total_frames},
        cluster_id=cluster_id,
    )
else:
    print("Reading cameraframes")
    cameraframes_df = retrieve_camera_frames(
        spark, moma_session_paths=[session_absolute_path], batch_size=BATCH_SIZE, zoom_level=zoom_level
    )
    frames_df = process_frames(
        cameraframes_df,
        rescale_size=INFERENCE_INPUT_IMG_SIZE,
        resize=INFERENCE_INPUT_IMG_SIZE != ZOOM_LEVEL_DICT[zoom_level],
    )
    print("Writing cameraframes")
    (
        frames_df.withColumn("run_id", F.lit(run_id))
        .write.format("delta")
        .mode("append")
        .partitionBy("session_name", "zoom_level")
        .save(PANORAMA_TABLE_PATH)
    )
    total_frames = (
        spark.read.format("delta")
        .load(PANORAMA_TABLE_PATH)
        .select("session_name")
        .filter(F.col("session_name") == session_name)
        .count()
    )
    print(f"Success: written {total_frames} frames for session {session_name}")
    
    write_metadata(
        spark,
        run_metadata_table_path=RUN_METADATA_TABLE_PATH,
        run_id=run_id,
        session_name=session_name,
        workflow="panorama",
        event="done",
        event_time=datetime.now(),
        databricks_job_run_id=databricks_job_run_id,
        databricks_job_parent_run_id=databricks_job_parent_run_id,
        input_parameters=input_parameters,
        panorama={"zoom_level": zoom_level, "total_frames": total_frames},
        cluster_id=cluster_id,
    )

# COMMAND ----------
