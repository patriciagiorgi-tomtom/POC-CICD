import pyspark
from databricks.sdk.runtime import dbutils
import logging
import sys
from datetime import datetime
import json

from databricks.sdk import WorkspaceClient
from databricks.sdk.service import jobs


def get_session_absolute_path_from_name(session_name: str, mount_destination: str, fullpath = False):
    if fullpath:
        year = session_name.split("_")[1]
        date_str = "_".join(session_name.split("_")[1:4])
        return f"{mount_destination}/{year}/{date_str}/{session_name}"
    else:
        return f"{mount_destination}/{session_name}"



def set_storage_account_config(
    spark: pyspark.sql.session.SparkSession,
    storage_account: str,
    client_id: str,
    tenant_id: str,
    service_credential: str,
):
    """Configures the spark session for a datalake connection

    Args:
        spark (pyspark.sql.session.SparkSession): Spark session being used
        storage_account (str): The storage account name being configured
        client_id (str): Client id for the datalake connection
        tenant_id (str): Tenant id for the datalake connection
        service_credential (_type_): The service credential created for the connection
    """
    # perform authentication
    spark.conf.set(f"fs.azure.account.auth.type.{storage_account}.dfs.core.windows.net", "OAuth")
    spark.conf.set(
        f"fs.azure.account.oauth.provider.type.{storage_account}.dfs.core.windows.net",
        "org.apache.hadoop.fs.azurebfs.oauth2.ClientCredsTokenProvider",
    )
    spark.conf.set(
        f"fs.azure.account.oauth2.client.id.{storage_account}.dfs.core.windows.net",
        client_id,
    )
    spark.conf.set(
        f"fs.azure.account.oauth2.client.secret.{storage_account}.dfs.core.windows.net",
        service_credential,
    )
    spark.conf.set(
        f"fs.azure.account.oauth2.client.endpoint.{storage_account}.dfs.core.windows.net",
        f"https://login.microsoftonline.com/{tenant_id}/oauth2/token",
    )


def mount_fs(
    spark: pyspark.sql.session.SparkSession,
    client_id: str,
    tenant_id: str,
    service_credential: str,
    mount_source: str,
    mount_destination: str,
):
    """Mount the directory requested

    Args:
        spark (pyspark.sql.session.SparkSession): Spark session being used
        client_id (str): Client id for the datalake connection
        tenant_id (str): Tenant id for the datalake connection
        service_credential (str): The service credential created for the connection
        mount_source (str): The source of the data to mount
        mount_destination (str): Destination of the mounting point

    Returns:
        pyspark.sql.session.SparkSession: Configured spark session
    """

    configs = {
        "fs.azure.account.auth.type": "OAuth",
        "fs.azure.account.oauth.provider.type": "org.apache.hadoop.fs.azurebfs.oauth2.ClientCredsTokenProvider",
        "fs.azure.account.oauth2.client.id": client_id,
        "fs.azure.account.oauth2.client.secret": service_credential,
        "fs.azure.account.oauth2.client.endpoint": f"https://login.microsoftonline.com/{tenant_id}/oauth2/token",
    }

    try:
        dbutils.fs.mount(source=mount_source, mount_point=mount_destination, extra_configs=configs)
    except Exception as e:
        if "Directory already mounted" not in e.__str__():
            raise RuntimeError("An Error occured while mounting the blob")


def setup_logging(name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    # # Create a StreamHandler instance that outputs to sys.stdout
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    return logger


def trigger_job(
    spark,
    run_metadata_table_path: str,
    run_id: str,
    job_id: str,
    job_parameters: dict,
    workflow: str,
    databricks_job_parent_run_id: str = None,
    realignment: dict = None,
    panorama: dict = None,
    inference: dict = None,
    lidar: dict = None,
    cluster_id=None,
):
    w = WorkspaceClient()
    print(f"Sending job {workflow} ({job_id})...")
    w.jobs.run_now(job_id=job_id, job_parameters=job_parameters)
    
    # write_metadata(
    #     spark,
    #     run_metadata_table_path=run_metadata_table_path,
    #     run_id=run_id,
    #     session_name=job_parameters["session_name"],
    #     workflow=workflow,
    #     event="trigger",
    #     event_time=datetime.now(),
    #     input_parameters=job_parameters,
    #     databricks_job_parent_run_id=databricks_job_parent_run_id,
    #     realignment=realignment,
    #     panorama=panorama,
    #     inference=inference,
    #     lidar=lidar,
    #     cluster_id=cluster_id,
    # )
    print("Job triggered and metadata written")


def write_metadata(
    spark: pyspark.sql.session.SparkSession,
    run_metadata_table_path: str,
    run_id: str,
    session_name: str,
    workflow: str,
    event: str,
    event_time: datetime,
    input_parameters: dict = None,
    databricks_job_run_id: str = None,
    databricks_job_parent_run_id: str = None,
    realignment: dict = None,
    panorama: dict = None,
    inference: dict = None,
    lidar: dict = None,
    cluster_id=None,
):
    w = WorkspaceClient()
    if cluster_id:
        cluster_info = w.clusters.get(cluster_id=cluster_id)
        databricks_cluster = {
            "driver_node_type_id": cluster_info.driver_node_type_id,
            "node_type_id": cluster_info.node_type_id,
            "num_workers": int(cluster_info.num_workers),
            "cluster_memory_mb": int(cluster_info.cluster_memory_mb),
            "cluster_cores": int(cluster_info.cluster_cores),
        }
    else:
        databricks_cluster = None

    spark.createDataFrame(
        [
            {
                "run_id": run_id,
                "session_name": session_name,
                "workflow": workflow,
                "event": event,
                "event_time": event_time,
                "input_parameters": json.dumps(input_parameters),
                "databricks_job_run_id": databricks_job_run_id,
                "databricks_job_parent_run_id": databricks_job_parent_run_id,
                "realignment": realignment,
                "panorama": panorama,
                "inference": inference,
                "lidar": lidar,
                "databricks_cluster": databricks_cluster,
            }
        ],
        schema="""
                run_id string, 
                session_name string, 
                workflow string,
                event string,
                event_time timestamp,
                input_parameters string,
                databricks_job_run_id string,
                databricks_job_parent_run_id string,
                realignment struct<
                    version string,
                    key string,
                    type string,
                    tsUrl string
                >,
                panorama struct<
                    zoom_level string,
                    total_frames integer
                >,
                inference struct<
                    mlflow_inference_run_id string,
                    mlflow_inference_experiment_id string,
                    mlflow_model_run_id string,
                    total_frames integer,
                    total_frames_with_inference integer,
                    total_inferences integer
                >,
                lidar struct<
                    lidar_frames_extracted integer
                >,
                databricks_cluster struct<
                    driver_node_type_id string,
                    node_type_id string,
                    num_workers integer,
                    cluster_memory_mb integer,
                    cluster_cores integer
                >
                """,
    ).write.format("delta").mode("append").save(run_metadata_table_path)
