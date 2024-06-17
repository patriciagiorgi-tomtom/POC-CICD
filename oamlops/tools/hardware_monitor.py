# Databricks notebook source
from oamlops.tools.monitoring.hardware_metric_processor import get_hardware_moniotring_metrics
from oamlops.utils.databricks_utils import set_storage_account_config

dbutils.widgets.text("storage_account", "reksiodbxstorageaccount")
storage_account = dbutils.widgets.get("storage_account")

tenant_id = dbutils.secrets.get(scope="repository-secrets", key="reksio-app-tenant-id")
principal_id = dbutils.secrets.get(scope="repository-secrets", key="reksio-app-principal-id")
principal_secret = dbutils.secrets.get(scope="repository-secrets", key="reksio-app-principal-secret")

set_storage_account_config(spark, storage_account, principal_id, tenant_id, principal_secret)

# COMMAND ----------

run_metadata_path = 'abfss://oamlops-bronze-dev@reksiodbxstorageaccount.dfs.core.windows.net/single-job-4/run_metadata.delta'  # "abfss://oamlops-bronze-dev@reksiodbxstorageaccount.dfs.core.windows.net/single-job-5/run_metadata.delta" #'abfss://oamlops-silver@reksiodbxstorageaccount.dfs.core.windows.net/dev/run_metadata.delta'
run_id = '2024-05-30/Ingolstad(all)/82'  # '2024-06-03/Ingolstad(all)/a1' #2024-05-28/Ingolstad2(all)/53'

# COMMAND ----------

test_df = get_hardware_moniotring_metrics(spark=spark, run_metadata_path=run_metadata_path, run_id=run_id)

# COMMAND ----------

test_df.display()

# COMMAND ----------

test_df.select('run_id', 'metric_key', 'value').groupby('run_id', 'metric_key').agg(F.mean('value')).display()
