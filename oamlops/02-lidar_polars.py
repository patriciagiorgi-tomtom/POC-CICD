# Databricks notebook source

# COMMAND ----------

# MAGIC %pip install --upgrade databricks-sdk

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

dbutils.widgets.text("session_name", "EL9AK18_2022_07_17__15_01_48")
dbutils.widgets.text("run_id", "")
dbutils.widgets.text("databricks_job_run_id", "")
dbutils.widgets.text("databricks_job_parent_run_id", "")


# COMMAND ----------

session_name = dbutils.widgets.get("session_name")
run_id = dbutils.widgets.get("run_id")
databricks_job_run_id = dbutils.widgets.get("databricks_job_run_id")
databricks_job_parent_run_id = dbutils.widgets.get("databricks_job_parent_run_id")

#get environment
dbutils.widgets.text("databricks_env", "") 
databricks_env =  dbutils.widgets.get("databricks_env") 

from config.settings import ConfigLoader
config_loader = ConfigLoader(databricks_env)
config = config_loader.get_config()
LIDAR_POLARS_TABLE_PATH = config['LIDAR_POLARS_TABLE_PATH']
LASER_NAME = config['LASER_NAME']
SQLITE_FOLDER_PATH = config['SQLITE_FOLDER_PATH']
MOUNT_DESTINATION = config['MOUNT_DESTINATION']
RUN_METADATA_TABLE_PATH = config['RUN_METADATA_TABLE_PATH']

input_parameters = {
    "session_name": session_name,
    "run_id": run_id,
    "databricks_job_run_id": databricks_job_run_id,
    "databricks_job_parent_run_id": databricks_job_parent_run_id,
    "run_metadata_table_path": RUN_METADATA_TABLE_PATH,
    "lidar_polars_table_path": LIDAR_POLARS_TABLE_PATH,
    "mount_source": SQLITE_FOLDER_PATH,
    "mount_destination": MOUNT_DESTINATION,
    "laser_name": LASER_NAME,
    "run_id": run_id,
}

cluster_id = dbutils.entry_point.getDbutils().notebook().getContext().tags().get("clusterId").get()

# COMMAND ----------


import comlink
import asyncio
import lzma
import numpy as np
import pandas as pd
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, ArrayType, IntegerType, ShortType
from data_loader.sqlite_data_provider import SqliteDataProvider
import equalaser
import os
from datetime import datetime
import nest_asyncio

nest_asyncio.apply()

from utils import databricks_utils
from utils.databricks_utils import write_metadata


logging = databricks_utils.setup_logging(
    os.path.basename(dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get())
)


# COMMAND ----------

tenant_id = dbutils.secrets.get(scope="repository-secrets", key="reksio-app-tenant-id")
principal_id = dbutils.secrets.get(scope="repository-secrets", key="reksio-app-principal-id")
principal_secret = dbutils.secrets.get(scope="repository-secrets", key="reksio-app-principal-secret")


storage_acc = LIDAR_POLARS_TABLE_PATH.split("@")[1].split(".")[0]
# Access to the tables with SP
databricks_utils.set_storage_account_config(spark, storage_acc, principal_id, tenant_id, principal_secret)
# Mount FS to access the SQLITE and LSQLITE
databricks_utils.mount_fs(spark, principal_id, tenant_id, principal_secret, SQLITE_FOLDER_PATH, MOUNT_DESTINATION)

# COMMAND ----------
from utils.databricks_utils import get_session_absolute_path_from_name

sqlite_path = rf"/dbfs{get_session_absolute_path_from_name(session_name, MOUNT_DESTINATION)}.SQLITE"
lsqlite_path = rf"/dbfs{get_session_absolute_path_from_name(session_name, MOUNT_DESTINATION)}.LSQLITE"

sdp = SqliteDataProvider(sqlite_path=sqlite_path, lsqlite_path=lsqlite_path)

# COMMAND ----------

write_metadata(
    spark,
    run_metadata_table_path=RUN_METADATA_TABLE_PATH,
    run_id=run_id,
    session_name=session_name,
    workflow="lidar_polars",
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

# MAGIC %md
# MAGIC ### Checking if session already processed

# COMMAND ----------

try:
     processed_session = (
        spark.read.format("delta").load(LIDAR_POLARS_TABLE_PATH).where(F.col("session_name") == session_name).count()
        > 0
    )
     if processed_session:
        dbutils.notebook.exit(0)
except:
    print("It looks like lidar_polars table does not exist. We are good to go.")

# COMMAND ----------

# MAGIC %md
# MAGIC # Saving polars

# COMMAND ----------

session = comlink.SessionName(session_name)
cam_calib = asyncio.run(comlink._panoramix.fetch_ladybug_calibration(session))
lcf = equalaser._calibration.LidarCalibrationFetcherWithExternal(ext_calibration_downloader=None)
sess_info = asyncio.run(equalaser._utils.SessInfo.fetch(session, calibration_fetcher=lcf))
time_info = sdp.get_time_index(laser_id=LASER_NAME)
DTYPE = np.dtype(
    [
        ("corresponding_blob_id", "<u2"),
        ("time_in_milliseconds", "<u4"),
        ("time_in_microseconds", "<u2"),
        ("corresponding_blob_offset", "<u4"),
    ]
)
time_records = np.frombuffer(lzma.decompress(time_info), dtype=DTYPE)
time_in_seconds = sess_info.scan_times

# COMMAND ----------


schema = StructType(
    [
        StructField("distance", ArrayType(IntegerType()), False),
        StructField("refl", ArrayType(ShortType()), False),
        StructField("theta", IntegerType(), False),
        StructField("time", DoubleType(), False),
        StructField("session_name", StringType(), False),
    ]
)


def uncompress_parse_polars(pdf):
    blob_id = pdf["idx"].iloc[0]
    blob = pdf["data"].iloc[0]
    session_name = pdf["session_name"].iloc[0]
    blob = lzma.decompress(blob)
    scan_times = time_in_seconds[time_records["corresponding_blob_id"] == blob_id]
    scans = equalaser._raster.parse_blob(blob, scan_times)
    return pd.DataFrame(
        data={
            "distance": scans["distance"].tolist(),
            "refl": scans["refl"].tolist(),
            "theta": scans["theta"],
            "time": scans["time"],
            "session_name": session_name,
        }
    )


laser_table = sdp.get_laser_data_table(LASER_NAME)

laser_data = (
    spark.read.format("jdbc").option("url", f"jdbc:sqlite:{lsqlite_path}").option("dbtable", laser_table).load()
)

total_blobs = laser_data.count()

# COMMAND ----------

laser_data = laser_data.withColumn("session_name", F.lit(session_name))
laser_data_uncompress = laser_data.groupby("idx", "data", "session_name").applyInPandas(
    uncompress_parse_polars, schema=schema
)
laser_data_uncompress.withColumn("run_id", F.lit(run_id))
# laser_data_uncompress.display()

# COMMAND ----------

laser_data_uncompress.write.format("delta").partitionBy("session_name").mode("append").save(LIDAR_POLARS_TABLE_PATH)

# COMMAND ----------

write_metadata(
    spark,
    run_metadata_table_path=RUN_METADATA_TABLE_PATH,
    run_id=run_id,
    session_name=session_name,
    workflow="lidar_polars",
    event="done",
    event_time=datetime.now(),
    databricks_job_run_id=databricks_job_run_id,
    databricks_job_parent_run_id=databricks_job_parent_run_id,
    input_parameters=input_parameters,
    cluster_id=cluster_id,
)

# spark.read.format("delta").load("abfss://oamlops-bronze@reksiodbxstorageaccount.dfs.core.windows.net/lidar_polars.delta").count()

# COMMAND ----------
