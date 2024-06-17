# Databricks notebook source
# DBTITLE 1,Install azcopy
# MAGIC %sh
# MAGIC
# MAGIC mkdir -p /tmp/moma_raw/
# MAGIC cd /tmp/moma_raw
# MAGIC
# MAGIC curl -L https://aka.ms/downloadazcopy-v10-linux | tar --strip-components=1 --exclude=*.txt -xzvf -; chmod +x azcopy
# MAGIC
# MAGIC ls
# MAGIC
# MAGIC echo "if you see azcopy in the line above, then azcopy installed correctly into /tmp/moma_raw/azcopy"

# COMMAND ----------

# MAGIC
# MAGIC %pip install --upgrade databricks-sdk

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,Constants (require commit to change)
# get environment
dbutils.widgets.text("databricks_env", "")
databricks_env = dbutils.widgets.get("databricks_env")

from config.settings import ConfigLoader

config_loader = ConfigLoader(databricks_env)
config = config_loader.get_config()

# COMMAND ----------

STORAGE_ACCOUNT = config['STORAGE_ACCOUNT']
SQLITE_FOLDER_PATH_HTTPS = config['SQLITE_FOLDER_PATH_HTTPS']
RUN_METADATA_TABLE_PATH = config['RUN_METADATA_TABLE_PATH']

cluster_id = dbutils.entry_point.getDbutils().notebook().getContext().tags().get("clusterId").get()

# COMMAND ----------

# DBTITLE 1,set credentials
from utils.databricks_utils import set_storage_account_config

tenant_id = dbutils.secrets.get(scope="repository-secrets", key="reksio-app-tenant-id")
principal_id = dbutils.secrets.get(scope="repository-secrets", key="reksio-app-principal-id")
principal_secret = dbutils.secrets.get(scope="repository-secrets", key="reksio-app-principal-secret")

set_storage_account_config(spark, STORAGE_ACCOUNT, principal_id, tenant_id, principal_secret)

# COMMAND ----------

# DBTITLE 1,Define inputs
dbutils.widgets.text("session_names", "EL2GM02_2023_07_28__20_01_50")
dbutils.widgets.text("session_names_table_path", "")
dbutils.widgets.text("destination_folder_sas_token", "")
dbutils.widgets.text("polygon_wkt", "")
dbutils.widgets.text("project_name", "test")
dbutils.widgets.text("inference.mlflow_model_run_id", "c4ddf6de80da4864bd85687aec176e26")
dbutils.widgets.text("panorama.zoom_level", "1")
dbutils.widgets.text("databricks_job_run_id", "")

dbutils.widgets.text("databricks_job_id_panorama", "")
dbutils.widgets.text("databricks_job_id_lidar_polars", "")
dbutils.widgets.text("databricks_job_id_inference", "")
dbutils.widgets.text("databricks_job_id_carpositions", "")
dbutils.widgets.text("databricks_job_id_trajectory", "")
dbutils.widgets.text("databricks_job_id_lidar_xyz", "")

# COMMAND ----------

# DBTITLE 1,Extract inputs
session_names = dbutils.widgets.get("session_names").split(",")
session_names_table_path = dbutils.widgets.get("session_names_table_path")
destination_folder_sas_token = dbutils.widgets.get("destination_folder_sas_token")
polygon_wkt = dbutils.widgets.get("polygon_wkt")
project_name = dbutils.widgets.get("project_name")
inference_mlflow_model_run_id = dbutils.widgets.get("inference.mlflow_model_run_id")
panorama_zoom_level = dbutils.widgets.get("panorama.zoom_level")
databricks_job_run_id = dbutils.widgets.get("databricks_job_run_id")

databricks_job_id_panorama = dbutils.widgets.get("databricks_job_id_panorama")
databricks_job_id_lidar_polars = dbutils.widgets.get("databricks_job_id_lidar_polars")
databricks_job_id_inference = dbutils.widgets.get("databricks_job_id_inference")
databricks_job_id_carpositions = dbutils.widgets.get("databricks_job_id_carpositions")
databricks_job_id_trajectory = dbutils.widgets.get("databricks_job_id_trajectory")
databricks_job_id_lidar_xyz = dbutils.widgets.get("databricks_job_id_lidar_xyz")

# COMMAND ----------

# DBTITLE 1,Parse polygon
from copy_sessions.copy_sessions import extract_sessions_from_polygon
import pandas

if len(polygon_wkt) > 0:
    print("Polygon has been supplied as input, using polygon to retrieve sessions")
    session_names = extract_sessions_from_polygon(spark, polygon_wkt)
elif session_names_table_path != "":
    print(f"Session names table path has been supplied as input, using table to retrieve sessions")
    session_df = pandas.read_csv(session_names_table_path).sample(frac=1, replace=False)
    session_names = list(session_df['session_name'].values)
elif len(session_names) > 0:
    print(f"Polygon not supplied. Using session_names ({len(session_names)}) to retrieve sessions")
else:
    raise AttributeError("Inputs session_names and polygon_wkt are empty. Not processing anything")

# COMMAND ----------

# DBTITLE 1,Generate run_id=date/project/random
import uuid
from datetime import datetime

run_id = f"{datetime.strftime(datetime.now(), '%Y-%m-%d')}/{project_name}/{str(uuid.uuid4())[:2]}"
print(f"Generated run_id='{run_id}'")

# COMMAND ----------

# DBTITLE 1,Filter sessions to get only realigned ones
from copy_sessions.copy_sessions import get_realigned_sessions, copy_sqlite

realigned_sessions = get_realigned_sessions(session_names).reset_index(drop=True)

print(f"Processing {len(session_names)} sessions")
realigned_sessions


# COMMAND ----------

def check_run_pipeline(group):
    processed_session = group.dropna(subset='session_name')
    row = group.iloc[0][["name", "key", "type", "version", "tsUrl"]]

    # Default execute to True
    row['run_carpositions'] = "yes"
    row['run_lidar_polars'] = "yes"
    row['run_panorama'] = "yes"
    row['run_inference'] = "yes"
    row['run_trajectory'] = "yes"
    row['run_lidar_xyz'] = "yes"
    if len(processed_session) == 0:
        return row

    # Carpositions
    carpositions = group[group['workflow'] == 'carpositions']
    if len(carpositions) > 0:
        row['run_carpositions'] = "no"

    # Lidar polars
    lidar_polars = group[group['workflow'] == 'lidar_polars']
    if len(lidar_polars) > 0:
        row['run_lidar_polars'] = "no"

    # Panorama
    panorama = group[group['workflow'] == 'panorama']
    for _, execution in panorama.iterrows():
        if (execution['panorama.zoom_level'] == panorama_zoom_level):
            row['run_panorama'] = "no"
            break

    # Trajectory
    trajectory = group[group['workflow'] == 'trajectory']
    if len(trajectory) > 0:
        for _, execution in trajectory.iterrows():
            if str(execution['version']) == str(execution['realignment.version']):
                row['run_trajectory'] = "no"
                break

    # Inference
    inference = group[group['workflow'] == 'inference']
    if len(inference) > 0:
        for _, execution in inference.iterrows():
            if (
                    (inference_mlflow_model_run_id == execution['mlflow_model_run_id'])
                    and (execution['panorama.zoom_level'] == panorama_zoom_level)
            ):
                row['run_inference'] = "no"
                break

    # Lidar XYZ
    lidar_xyz = group[group['workflow'] == 'lidar_xyz']
    if len(lidar_xyz) > 0:
        for _, execution in lidar_xyz.iterrows():
            if (
                    (str(execution['version']) == str(execution['realignment.version']))
                    and (inference_mlflow_model_run_id == execution['mlflow_model_run_id'])
            ):
                row['run_lidar_xyz'] = "no"
                break

    return row


# COMMAND ----------

from datetime import datetime
import time
import random

from utils.databricks_utils import write_metadata, trigger_job
from copy_sessions.copy_sessions import get_workflows_to_run
import pyspark.sql.functions as F

from databricks.sdk import WorkspaceClient

try:
    metadata = spark.read.format("delta").load(RUN_METADATA_TABLE_PATH)
    metadata_processed = (
        metadata.filter(F.col("event") == "done")
        .select(
            "session_name",
            "workflow",
            F.col("panorama.zoom_level").alias("panorama.zoom_level"),
            F.col("realignment.version").alias("realignment.version"),
            F.col("inference.mlflow_model_run_id").alias("mlflow_model_run_id")
        )
    ).toPandas()

    realigned_sessions = realigned_sessions.merge(metadata_processed, left_on='name', right_on='session_name',
                                                  how='left').groupby('name').apply(check_run_pipeline)

except:
    print("Metadata table not found...")
    metadata = None
    realigned_sessions['run_carpositions'] = "yes"
    realigned_sessions['run_lidar_polars'] = "yes"
    realigned_sessions['run_panorama'] = "yes"
    realigned_sessions['run_inference'] = "yes"
    realigned_sessions['run_trajectory'] = "yes"
    realigned_sessions['run_lidar_xyz'] = "yes"
    
    
realigned_sessions = realigned_sessions.reset_index(drop=True)

for i, row in realigned_sessions.iterrows():

    session_name = row["name"]
    realignment_version = row["version"]
    realignment_key = row["key"]
    realignment_type = row["type"]
    realignment_tsUrl = row["tsUrl"]

    realignment = {
        "version": realignment_version,
        "key": realignment_key,
        "type": realignment_type,
        "tsUrl": realignment_tsUrl,
    }

    run_panorama = "yes"
    run_trajectory = "yes"
    run_carpositions = "yes"
    run_lidar_polars = "yes"
    run_lidar_xyz = "yes"
    run_inference = "yes"

    print(f"progress: {i}/{len(realigned_sessions)}: {session_name}")
                    
    if any([run_carpositions=="yes", run_trajectory=="yes", run_panorama=="yes", run_inference=="yes", run_lidar_polars=="yes", run_lidar_xyz=="yes"]):
            
        status = copy_sqlite(session_name = session_name, destination_folder=SQLITE_FOLDER_PATH_HTTPS, destination_folder_sas_token=destination_folder_sas_token)
        if status != 0:
            print("ERROR: Skipping this session. Please, check it later!")
            continue
        
        databricks_jobs_to_run = []
        if run_panorama=="yes":
            print("Running panorama")
            databricks_jobs_to_run.append({"job_id": databricks_job_id_panorama, "workflow": "panorama"})
        else:
            
            if run_inference=="yes":
                print("Running inference")
                databricks_jobs_to_run.append({"job_id": databricks_job_id_inference, "workflow": "inference"})
            elif run_lidar_polars=="no":
                print("Running lidar_xyz")
                databricks_jobs_to_run.append({"job_id": databricks_job_id_lidar_xyz, "workflow": "lidar_xyz"})
            
            if run_carpositions=="yes":
                print("Running carpositions")
                databricks_jobs_to_run.append({"job_id": databricks_job_id_carpositions, "workflow": "carpositions"})
            elif run_trajectory=="yes":
                print("Running trajectory")
                databricks_jobs_to_run.append({"job_id": databricks_job_id_trajectory, "workflow": "trajectory"})
            elif run_lidar_polars=="yes":
                print("Running lidar_polars")
                databricks_jobs_to_run.append({"job_id": databricks_job_id_lidar_polars, "workflow": "lidar_polars"})
        
        for job_specs in databricks_jobs_to_run:
            
            # if random.random() < 0.05: # 5% of the time , stochastic check
            #     print("Checking if there are any queued runs for this job...")
            #     w = WorkspaceClient()
            #     queue_size = 999
            #     while queue_size > 1:
            #         queue_size = len([1 for x in w.jobs.list_runs(job_id=job_specs['job_id'], active_only=True) if str(x.state.life_cycle_state) == 'RunLifeCycleState.QUEUED'])
            #         print(f"There are {queue_size} queued runs for this job.")
            #         if queue_size > 1:
            #             print("Sleeping for 60 seconds...")
            #             time.sleep(60)
            
            trigger_job(
                    spark,
                    run_metadata_table_path=RUN_METADATA_TABLE_PATH,
                    run_id=run_id,
                    job_id=job_specs['job_id'],
                    job_parameters={
                        'databricks_env':databricks_env,#giorgip
                        "session_name": session_name,
                        "run_id": run_id,
                        "realignment.version": realignment_version,
                        "realignment.key": realignment_key,
                        "realignment.type": realignment_type,
                        "realignment.tsUrl": realignment_tsUrl,
                        "databricks_job_parent_run_id": databricks_job_run_id,
                        "panorama.zoom_level": panorama_zoom_level,
                        "inference.mlflow_model_run_id": inference_mlflow_model_run_id,
                        "run_trajectory": run_trajectory,
                        "run_panorama": run_panorama,
                        "run_carpositions": run_carpositions,
                        "run_inference": run_inference,
                        "run_lidar_polars": run_lidar_polars,
                        "run_lidar_xyz": run_lidar_xyz,
                        "databricks_job_id_panorama": databricks_job_id_panorama,
                        "databricks_job_id_lidar_polars": databricks_job_id_lidar_polars,
                        "databricks_job_id_inference": databricks_job_id_inference,
                        "databricks_job_id_carpositions": databricks_job_id_carpositions,
                        "databricks_job_id_trajectory": databricks_job_id_trajectory,
                        "databricks_job_id_lidar_xyz": databricks_job_id_lidar_xyz,
                    },
                    workflow=job_specs["workflow"],
                    databricks_job_parent_run_id=databricks_job_run_id,
                    realignment=realignment,
                    cluster_id=cluster_id,
                )

# COMMAND ----------

import pyspark.sql.functions as F

spark.read.format("delta").load(RUN_METADATA_TABLE_PATH).orderBy(F.desc("event_time")).display()
