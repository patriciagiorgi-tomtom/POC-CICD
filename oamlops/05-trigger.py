# Databricks notebook source

# COMMAND ----------

# MAGIC %pip install --upgrade databricks-sdk

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

cluster_id = (
    dbutils.entry_point.getDbutils()
    .notebook()
    .getContext()
    .tags()
    .get("clusterId")
    .get()
)


# COMMAND ----------

dbutils.widgets.text("databricks_env", "")

dbutils.widgets.text("databricks_job_id_panorama", "")
dbutils.widgets.text("databricks_job_id_lidar_polars", "")
dbutils.widgets.text("databricks_job_id_inference", "")
dbutils.widgets.text("databricks_job_id_carpositions", "")
dbutils.widgets.text("databricks_job_id_trajectory", "")
dbutils.widgets.text("databricks_job_id_lidar_xyz", "")

dbutils.widgets.text("next_stages", "")
dbutils.widgets.text("blocker_stages", "")

dbutils.widgets.text("run_id", "")

dbutils.widgets.text("session_name", "EL2GM02_2023_07_28__20_01_50")
dbutils.widgets.text("panorama.zoom_level", "1")

dbutils.widgets.text("realignment.version", "")
dbutils.widgets.text("realignment.key", "")
dbutils.widgets.text("realignment.type", "")
dbutils.widgets.text("realignment.tsUrl", "")
dbutils.widgets.text("inference.mlflow_model_run_id", "")

dbutils.widgets.text("databricks_job_run_id", "")
dbutils.widgets.text("databricks_job_parent_run_id", "")

dbutils.widgets.text("run_panorama", "yes")
dbutils.widgets.text("run_carpositions", "yes")
dbutils.widgets.text("run_inference", "yes")
dbutils.widgets.text("run_lidar_polars", "yes")
dbutils.widgets.text("run_trajectory", "yes")
dbutils.widgets.text("run_lidar_xyz", "yes")

# COMMAND ----------


databricks_env = dbutils.widgets.get("databricks_env")

databricks_job_id_panorama = dbutils.widgets.get("databricks_job_id_panorama")
databricks_job_id_lidar_polars = dbutils.widgets.get("databricks_job_id_lidar_polars")
databricks_job_id_inference = dbutils.widgets.get("databricks_job_id_inference")
databricks_job_id_carpositions = dbutils.widgets.get("databricks_job_id_carpositions")
databricks_job_id_trajectory = dbutils.widgets.get("databricks_job_id_trajectory")
databricks_job_id_lidar_xyz = dbutils.widgets.get("databricks_job_id_lidar_xyz")

next_stages = dbutils.widgets.get("next_stages").split(",")
blocker_stages = dbutils.widgets.get("blocker_stages").split(",")

run_id = dbutils.widgets.get("run_id")

zoom_level = int(dbutils.widgets.get("panorama.zoom_level"))
session_name = dbutils.widgets.get("session_name")
databricks_job_run_id = dbutils.widgets.get("databricks_job_run_id")
databricks_job_parent_run_id = dbutils.widgets.get("databricks_job_parent_run_id")

realignment_version = dbutils.widgets.get("realignment.version")
realignment_key = dbutils.widgets.get("realignment.key")
realignment_type = dbutils.widgets.get("realignment.type")
realignment_tsUrl = dbutils.widgets.get("realignment.tsUrl")

inference_mlflow_model_run_id = dbutils.widgets.get("inference.mlflow_model_run_id")

run_panorama = dbutils.widgets.get("run_panorama")
run_lidar_polars = dbutils.widgets.get("run_lidar_polars")
run_inference = dbutils.widgets.get("run_inference")
run_carpositions = dbutils.widgets.get("run_carpositions")
run_trajectory = dbutils.widgets.get("run_trajectory")
run_lidar_xyz = dbutils.widgets.get("run_lidar_xyz")

# COMMAND ----------

from config.settings import ConfigLoader

config_loader = ConfigLoader(databricks_env)
config = config_loader.get_config()

RUN_METADATA_TABLE_PATH = config["RUN_METADATA_TABLE_PATH"]
STORAGE_ACCOUNT = config["STORAGE_ACCOUNT"]

# COMMAND ----------

# DBTITLE 1,set credentials
from utils.databricks_utils import set_storage_account_config

tenant_id = dbutils.secrets.get(scope="repository-secrets", key="reksio-app-tenant-id")
principal_id = dbutils.secrets.get(scope="repository-secrets", key="reksio-app-principal-id")
principal_secret = dbutils.secrets.get(scope="repository-secrets", key="reksio-app-principal-secret")

set_storage_account_config(spark, STORAGE_ACCOUNT, principal_id, tenant_id, principal_secret)

# COMMAND ----------

from utils.databricks_utils import trigger_job

# COMMAND ----------
import pyspark.sql.functions as F

for blocker_stage in blocker_stages:
    if blocker_stage == "":
        continue
    
    if (
            spark.read.format("delta").load(RUN_METADATA_TABLE_PATH)
            .filter(F.col("run_id") == run_id)
            .filter(F.col("session_name") == session_name)
            .filter(F.col("workflow") == blocker_stage)
            .filter(F.col("event").isin(["done", "skipped"]))
            .count()
    ) == 0:
        dbutils.notebook.exit(f"Blocker stage {blocker_stage} not done")

# COMMAND ----------

for next_stage in next_stages:
    if next_stage == "panorama":
        next_job_id = databricks_job_id_panorama
    elif next_stage == "lidar_polars":
        next_job_id = databricks_job_id_lidar_polars
    elif next_stage == "inference":
        next_job_id = databricks_job_id_inference
    elif next_stage == "carpositions":
        next_job_id = databricks_job_id_carpositions
    elif next_stage == "trajectory":
        next_job_id = databricks_job_id_trajectory
    elif next_stage == "lidar_xyz":
        next_job_id = databricks_job_id_lidar_xyz
    else:
        print(f"Invalid next_stage: {next_stage}")
        continue

    trigger_job(
        spark,
        run_metadata_table_path=RUN_METADATA_TABLE_PATH,
        run_id=run_id,
        job_id=next_job_id,
        job_parameters={
            "databricks_env": databricks_env,  # giorgip
            "session_name": session_name,
            "run_id": run_id,
            "realignment.version": realignment_version,
            "realignment.key": realignment_key,
            "realignment.type": realignment_type,
            "realignment.tsUrl": realignment_tsUrl,
            "databricks_job_parent_run_id": databricks_job_run_id,
            "panorama.zoom_level": zoom_level,
            "inference.mlflow_model_run_id": inference_mlflow_model_run_id,
            "databricks_job_id_panorama": databricks_job_id_panorama,
            "databricks_job_id_lidar_polars": databricks_job_id_lidar_polars,
            "databricks_job_id_inference": databricks_job_id_inference,
            "databricks_job_id_carpositions": databricks_job_id_carpositions,
            "databricks_job_id_trajectory": databricks_job_id_trajectory,
            "databricks_job_id_lidar_xyz": databricks_job_id_lidar_xyz,
            "run_panorama": run_panorama,
            "run_lidar_polars": run_lidar_polars,
            "run_inference": run_inference,
            "run_carpositions": run_carpositions,
            "run_trajectory": run_trajectory,
            "run_lidar_xyz": run_lidar_xyz,
        },
        workflow=next_stage,
        databricks_job_parent_run_id=databricks_job_run_id,
        cluster_id=cluster_id,
    )
