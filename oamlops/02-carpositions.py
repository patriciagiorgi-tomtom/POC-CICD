# Databricks notebook source
# %pip install /dbfs/FileStore/ADSENSING-python-dependencies/moma_sqlite_data_handlers-2023.9.1-py3-none-any.whl
# %pip install shapely tqdm h3
# MAGIC %pip install --upgrade databricks-sdk

dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,Constants
#get environment
dbutils.widgets.text("databricks_env", "") 
databricks_env =  dbutils.widgets.get("databricks_env") 

from config.settings import ConfigLoader
config_loader = ConfigLoader(databricks_env)
config = config_loader.get_config()
STORAGE_ACCOUNT = config['STORAGE_ACCOUNT']
RUN_METADATA_TABLE_PATH = config['RUN_METADATA_TABLE_PATH']
CARPOSITIONS_TABLE_PATH = config['CARPOSITIONS_TABLE_PATH']
SQLITE_FOLDER_PATH = config['SQLITE_FOLDER_PATH']
MOUNT_DESTINATION = config['MOUNT_DESTINATION']


cluster_id = dbutils.entry_point.getDbutils().notebook().getContext().tags().get("clusterId").get()

# COMMAND ----------

# DBTITLE 1,Set credentials
from utils.databricks_utils import set_storage_account_config, write_metadata

tenant_id = dbutils.secrets.get(scope="repository-secrets", key="reksio-app-tenant-id")
principal_id = dbutils.secrets.get(scope="repository-secrets", key="reksio-app-principal-id")
principal_secret = dbutils.secrets.get(scope="repository-secrets", key="reksio-app-principal-secret")

set_storage_account_config(spark, STORAGE_ACCOUNT, principal_id, tenant_id, principal_secret)

# COMMAND ----------

# DBTITLE 1,Mount SQLITE Path
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
    if "Directory already mounted" not in e.__str__():
        raise RuntimeError("An Error occured while mounting the blob")

# COMMAND ----------

# DBTITLE 1,Inputs
dbutils.widgets.text("session_name", "EL2GM02_2023_07_28__20_01_50")
dbutils.widgets.text("databricks_job_run_id", "")
dbutils.widgets.text("databricks_job_parent_run_id", "")
dbutils.widgets.text("run_id", "")

# COMMAND ----------

# DBTITLE 1,Parse inputs
session_name = dbutils.widgets.get("session_name")
databricks_job_run_id = dbutils.widgets.get("databricks_job_run_id")
databricks_job_parent_run_id = dbutils.widgets.get("databricks_job_parent_run_id")
run_id = dbutils.widgets.get("run_id")

input_parameters = {
    "session_name": session_name,
    "databricks_job_run_id": databricks_job_run_id,
    "databricks_job_parent_run_id": databricks_job_parent_run_id,
    "run_id": run_id,
    "CARPOSITIONS_TABLE_PATH": CARPOSITIONS_TABLE_PATH,
    "STORAGE_ACCOUNT": STORAGE_ACCOUNT,
    "SQLITE_FOLDER_PATH": SQLITE_FOLDER_PATH,
    "MOUNT_DESTINATION": MOUNT_DESTINATION,
    "RUN_METADATA_TABLE_PATH": RUN_METADATA_TABLE_PATH,
}


# COMMAND ----------
from utils.databricks_utils import get_session_absolute_path_from_name

session_absolute_path = "/dbfs" + get_session_absolute_path_from_name(session_name, MOUNT_DESTINATION) + ".SQLITE"

print(f"Starting run_id '{run_id}' for session_name '{session_name}' from {session_absolute_path}")

from datetime import datetime

write_metadata(
    spark,
    run_metadata_table_path=RUN_METADATA_TABLE_PATH,
    run_id=run_id,
    session_name=session_name,
    workflow="carpositions",
    event="started",
    event_time=datetime.now(),
    input_parameters=input_parameters,
    databricks_job_run_id=databricks_job_run_id,
    databricks_job_parent_run_id=databricks_job_parent_run_id,
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

import pyspark.sql.functions as F
from process_panoramas.sqlite_to_delta import retrieve_car_positions_for_sessions
from utils.databricks_utils import write_metadata
from datetime import datetime

print("Checking if session_name already exists in carpositions")
try:
    prev_carpositions = (
        spark.read.format("delta")
        .load(CARPOSITIONS_TABLE_PATH)
        .select("session_name")
        .filter(F.col("session_name") == session_name)
        .count()
    )
except:
    print("Looks like carposition table does not exist, we are good to write")
    prev_carpositions = 0

if prev_carpositions > 0:
    print(f"Session_name '{session_name}' already exists in carpositions. We are skipping this one...")
    write_metadata(
        spark,
        run_metadata_table_path=RUN_METADATA_TABLE_PATH,
        run_id=run_id,
        session_name=session_name,
        workflow="carpositions",
        event="skipped",
        event_time=datetime.now(),
        input_parameters=input_parameters,
        databricks_job_run_id=databricks_job_run_id,
        databricks_job_parent_run_id=databricks_job_parent_run_id,
        cluster_id=cluster_id,
    )
else:
    print("Reading car positions")
    carpositions_df = retrieve_car_positions_for_sessions(spark, [session_absolute_path])

    print("Writing car positions")
    (
        carpositions_df.withColumn("run_id", F.lit(run_id))
        .write.format("delta")
        .mode("append")
        .partitionBy("session_name")
        .save(CARPOSITIONS_TABLE_PATH)
    )
    print(f"Success: written {carpositions_df.count()} positions for session {session_name}")
    write_metadata(
        spark,
        run_metadata_table_path=RUN_METADATA_TABLE_PATH,
        run_id=run_id,
        session_name=session_name,
        workflow="carpositions",
        event="done",
        event_time=datetime.now(),
        input_parameters=input_parameters,
        databricks_job_run_id=databricks_job_run_id,
        databricks_job_parent_run_id=databricks_job_parent_run_id,
        cluster_id=cluster_id,
    )

# COMMAND ----------

# spark.read.format("delta").load(RUN_METADATA_TABLE_PATH).orderBy(F.desc("event_time")).display()

# COMMAND ----------
