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


# MAGIC %pip install --upgrade databricks-sdk

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,Constants (require commit to change)
#get environment
dbutils.widgets.text("databricks_env", "") 
databricks_env =  dbutils.widgets.get("databricks_env") 

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

session_df = pandas.read_csv('copy_sessions/detroit.csv').sample(frac=1, replace=False)
session_names = list(session_df['session_name'].values)

if len(polygon_wkt) > 0:
    print("Polygon has been supplied as input, using polygon to retrieve sessions")
    session_names = extract_sessions_from_polygon(spark, polygon_wkt)
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

def check_run_pipeline(row):
    processed_session = not pandas.isna(row['session_name'])
    
    # Check 
    # Carpositions
    # - session_name
    # Lidar Polars
    # - session_name
    # Panorama
    # - session_name
    # Inference
    # - session_name
    # - model_mlflow_run_id
    # Trajectory
    # - session_name
    # - realignment_version
    # LidarXYZ
    # - session_name
    # - model_mlflow_run_id
    # - realignment_version
    
    # Default execute to True
    row['run_carpositions'] = True
    row['run_lidar_polars'] = True
    row['run_panorama'] = True
    row['run_inference'] = True
    row['run_trajectory'] = True
    row['run_lidar_xyz'] = True
    if not processed_session:
        return row
    
    # Session is processed but need to check what is executed already
    if len(row['completed_model_run_id'])>0:
        model_run_id_processed = inference_mlflow_model_run_id in row['completed_model_run_id']
    else:
        model_run_id_processed = False
    if len(row['completed_realignments'])>0:
        realignment_processed = row['version'] in row['completed_realignments'].astype(int)
    else:
        realignment_processed = False

    if "carpositions" in row["completed_workflows"]:
        row['run_carpositions'] = False
    if "lidar_polars" in row["completed_workflows"]:
        row['run_lidar_polars'] = False


    if "panorama" in row["completed_workflows"]: # TODO add zoom
        row['run_panorama'] = False
    
    if "inference" in row["completed_workflows"] and (model_run_id_processed):
        row['run_inference'] = False
    if "trajectory" in row["completed_workflows"] and (realignment_processed):
        row['run_trajectory'] = False
    if "lidar_xyz" in row["completed_workflows"] and (model_run_id_processed) and (realignment_processed):
        row['run_lidar_xyz'] = False

    return row

# COMMAND ----------

from datetime import datetime
import time

from utils.databricks_utils import write_metadata, trigger_job
from copy_sessions.copy_sessions import get_workflows_to_run
import pyspark.sql.functions as F

from databricks.sdk import WorkspaceClient

try:
    metadata = spark.read.format("delta").load(RUN_METADATA_TABLE_PATH)
    metadata_processed = (
        metadata.where(F.col("event")=="done").groupBy("session_name").agg(
            F.collect_set("workflow").alias("completed_workflows"), 
            F.collect_set("realignment.version").alias("completed_realignments"),
            F.collect_set("inference.mlflow_model_run_id").alias("completed_model_run_id"))
    ).toPandas()

    realigned_sessions = realigned_sessions.merge(metadata_processed, left_on='name', right_on='session_name', how='left')
    realigned_sessions["completed_model_run_id"] = realigned_sessions["completed_model_run_id"].apply(lambda x: [] if pandas.isna(x) else x)
    realigned_sessions["completed_realignments"] = realigned_sessions["completed_realignments"].apply(lambda x: [] if pandas.isna(x) else x)
    realigned_sessions = realigned_sessions.apply(check_run_pipeline, axis=1)
    
except:
    print("Metadata table not found...")
    metadata = None
    

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
    
    run_panorama = row['run_panorama']
    run_trajectory = row['run_trajectory']
    run_carpositions = row['run_carpositions']
    run_lidar_polars = row['run_lidar_polars']
    run_lidar_xyz = row['run_lidar_xyz']
    run_inference = row['run_inference']

    print(f"progress: {i}/{len(realigned_sessions)}: {session_name}")
    
    #if any([run_carpositions, run_trajectory, run_panorama, run_inference, run_lidar_polars, run_lidar_xyz]):
    if any([run_carpositions, run_trajectory, run_panorama]):
        
        run_panorama = "yes" if row['run_panorama'] else "no"
        run_trajectory = "yes" if row['run_trajectory'] else "no"
        run_carpositions = "yes" if row['run_carpositions'] else "no"
        run_lidar_polars = "yes" if row['run_lidar_polars'] else "no"
        run_lidar_xyz = "yes" if row['run_lidar_xyz'] else "no"
        run_inference = "yes" if row['run_inference'] else "no"
        
        status = copy_sqlite(session_name = session_name, destination_folder=SQLITE_FOLDER_PATH_HTTPS, destination_folder_sas_token=destination_folder_sas_token)
        if status != 0:
            print("ERROR: Skipping this session. Please, check it later!")
            continue
        
        if i % 100 == 0:
            print("Checking if there are any queued runs for this job...")
            w = WorkspaceClient()
            queue_size = 999
            while queue_size > 20:
                queue_size = len([1 for x in w.jobs.list_runs(job_id=databricks_job_id_panorama, active_only=True) if str(x.state.life_cycle_state) == 'RunLifeCycleState.QUEUED'])
                print(f"There are {queue_size} queued runs for this job.")
                if queue_size > 1:
                    print("Sleeping for 60 seconds...")
                    time.sleep(60)
        
        trigger_job(
                spark,
                run_metadata_table_path=RUN_METADATA_TABLE_PATH,
                run_id=run_id,
                job_id=databricks_job_id_panorama,
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
                workflow="panorama",
                databricks_job_parent_run_id=databricks_job_run_id,
                realignment=realignment,
                cluster_id=cluster_id,
            )

# COMMAND ----------

import pyspark.sql.functions as F

spark.read.format("delta").load(RUN_METADATA_TABLE_PATH).orderBy(F.desc("event_time")).display()

# COMMAND ----------
