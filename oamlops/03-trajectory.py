# Databricks notebook source
# %pip install /dbfs/FileStore/ADSENSING-python-dependencies/moma_sqlite_data_handlers-2023.9.1-py3-none-any.whl
# %pip install /dbfs/FileStore/ADSENSING-python-dependencies/ad_aiotools-0.4.3-py3-none-any.whl
# %pip install /dbfs/FileStore/ADSENSING-python-dependencies/acircuit-0.3.3-py3-none-any.whl
# %pip install /dbfs/FileStore/ADSENSING-python-dependencies/comlink-0.7.2-py3-none-any.whl
# %pip install /dbfs/FileStore/ADSENSING-python-dependencies/equalaser-1.5.1-cp310-cp310-linux_x86_64.whl

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
TRAJECTORY_TABLE_PATH = config['TRAJECTORY_TABLE_PATH']
CARPOSITIONS_TABLE_PATH = config['CARPOSITIONS_TABLE_PATH']
RUN_METADATA_TABLE_PATH = config['RUN_METADATA_TABLE_PATH']
AZURE_LIDAR_CALIBRATION_SAS = config['AZURE_LIDAR_CALIBRATION_SAS']


cluster_id = dbutils.entry_point.getDbutils().notebook().getContext().tags().get("clusterId").get()

# COMMAND ----------

# DBTITLE 1,Set credentials
from utils.databricks_utils import set_storage_account_config, write_metadata

tenant_id = dbutils.secrets.get(scope="repository-secrets", key="reksio-app-tenant-id")
principal_id = dbutils.secrets.get(scope="repository-secrets", key="reksio-app-principal-id")
principal_secret = dbutils.secrets.get(scope="repository-secrets", key="reksio-app-principal-secret")

set_storage_account_config(spark, STORAGE_ACCOUNT, principal_id, tenant_id, principal_secret)

# COMMAND ----------

dbutils.widgets.text("session_name", "EL2GM02_2023_07_28__20_01_50")

dbutils.widgets.text("realignment.version", "0")
dbutils.widgets.text("realignment.key", "trajectory")
dbutils.widgets.text("realignment.type", "s2s")
dbutils.widgets.text("realignment.tsUrl", "http://service.trajectory-store-prod.tthad.net/trajectory-store")

dbutils.widgets.text("databricks_job_run_id", "")
dbutils.widgets.text("databricks_job_parent_run_id", "")
dbutils.widgets.text("run_id", "")

# COMMAND ----------

session_name = dbutils.widgets.get("session_name")

realignment_version = dbutils.widgets.get("realignment.version")
realignment_key = dbutils.widgets.get("realignment.key")
realignment_type = dbutils.widgets.get("realignment.type")
realignment_tsUrl = dbutils.widgets.get("realignment.tsUrl")

databricks_job_run_id = dbutils.widgets.get("databricks_job_run_id")
databricks_job_parent_run_id = dbutils.widgets.get("databricks_job_parent_run_id")
run_id = dbutils.widgets.get("run_id")

input_parameters = {
    "session_name": session_name,
    "realignment.version": realignment_version,
    "realignment.key": realignment_key,
    "realignment.type": realignment_type,
    "realignment.tsUrl": realignment_tsUrl,
    "databricks_job_run_id": databricks_job_run_id,
    "databricks_job_parent_run_id": databricks_job_parent_run_id,
    "run_id": run_id,
    "CARPOSITIONS_TABLE_PATH": CARPOSITIONS_TABLE_PATH,
    "STORAGE_ACCOUNT": STORAGE_ACCOUNT,
    "RUN_METADATA_TABLE_PATH": RUN_METADATA_TABLE_PATH,
}

realignment = {
    "version": realignment_version,
    "key": realignment_key,
    "type": realignment_type,
    "tsUrl": realignment_tsUrl,
    }

print(
    f"Processing session: '{session_name}', run_id:'{run_id}', databricks_job_run_id:'{databricks_job_run_id}', databricks_job_parent_run_id: '{databricks_job_parent_run_id}'"
)

from datetime import datetime

write_metadata(
    spark,
    run_metadata_table_path=RUN_METADATA_TABLE_PATH,
    run_id=run_id,
    session_name=session_name,
    workflow="trajectory",
    event="started",
    event_time=datetime.now(),
    input_parameters=input_parameters,
    databricks_job_run_id=databricks_job_run_id,
    databricks_job_parent_run_id=databricks_job_parent_run_id,
    cluster_id=cluster_id,
)

#  # COMMAND ----------
# year = session_name.split("_")[1]
# date_str = "_".join(session_name.split("_")[1:4])

# # COMMAND ----------
# %ls /dbfs{MOUNT_DESTINATION}

# # COMMAND ----------
# %ls /dbfs{MOUNT_DESTINATION}/{year}

# # COMMAND ----------
# %ls /dbfs{MOUNT_DESTINATION}/{year}/{date_str}


# COMMAND ----------
import os
from pyspark.sql import functions as F
from datetime import datetime
from utils import databricks_utils

import nest_asyncio

nest_asyncio.apply()

from oamlops.trajectory.trajectory import process_session_trajectory

logging = databricks_utils.setup_logging(
    os.path.basename(dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get())
)

# COMMAND ----------

from utils.databricks_utils import write_metadata

try:
    trajectories_table = (
        spark.read.format("delta").load(TRAJECTORY_TABLE_PATH).select("session_name", "realignment_version")
    )
    if (
        trajectories_table.filter(trajectories_table.session_name == session_name)
        .filter(trajectories_table.realignment_version == realignment_version)
        .count()
        > 0
    ):
        logging.info("Trajectory already processed")
        write_metadata(
            spark,
            run_metadata_table_path=RUN_METADATA_TABLE_PATH,
            run_id=run_id,
            session_name=session_name,
            workflow="trajectory",
            event="skipped",
            event_time=datetime.now(),
            input_parameters=input_parameters,
            databricks_job_run_id=databricks_job_run_id,
            databricks_job_parent_run_id=databricks_job_parent_run_id,
            cluster_id=cluster_id,
        )

        dbutils.notebook.exit(0)

except Exception as ex:
    logging.info(f"Extracting trajectory, ex: {ex}")

# COMMAND ----------

# DBTITLE 1,Write trajectories
(
    process_session_trajectory(
        spark,
        session_name,
        CARPOSITIONS_TABLE_PATH,
        realignment_version,
        realignment_key,
        realignment_type,
        realignment_tsUrl,
        AZURE_LIDAR_CALIBRATION_SAS,
    )
    .withColumn("realignment_version", F.lit(realignment_version))
    .withColumn("run_id", F.lit(run_id))
    .write.format("delta")
    .mode("append")
    .partitionBy("session_name")
    .save(TRAJECTORY_TABLE_PATH)
)

# COMMAND ----------

# DBTITLE 1,Write metadata
write_metadata(
    spark,
    run_metadata_table_path=RUN_METADATA_TABLE_PATH,
    run_id=run_id,
    session_name=session_name,
    workflow="trajectory",
    event="done",
    event_time=datetime.now(),
    input_parameters=input_parameters,
    databricks_job_run_id=databricks_job_run_id,
    databricks_job_parent_run_id=databricks_job_parent_run_id,
    realignment=realignment,
    cluster_id=cluster_id,
)

# COMMAND ----------

# spark.read.format("delta").load(RUN_METADATA_TABLE_PATH).orderBy(F.desc("event_time")).display()
