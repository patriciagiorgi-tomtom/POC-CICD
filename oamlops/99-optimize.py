# Databricks notebook source

# COMMAND ----------

# MAGIC %pip install --upgrade databricks-sdk

# COMMAND ----------

dbutils.library.restartPython()


# COMMAND ----------

from delta.tables import DeltaTable


# COMMAND ----------

def optimize_delta(delta_path):
    delta_table = DeltaTable.forPath(spark, delta_path)
        # Optimize the Delta table
    delta_table.optimize().executeCompaction()
    spark.conf.set("spark.databricks.delta.retentionDurationCheck.enabled", "false")
    delta_table.vacuum(168)  # 168 hours = 7 days


# COMMAND ----------

dbutils.widgets.text("databricks_env", "") 
databricks_env =  dbutils.widgets.get("databricks_env") 

from config.settings import ConfigLoader
config_loader = ConfigLoader(databricks_env)
config = config_loader.get_config()

tables_to_optimize = [key for key in config.keys() if 'TABLE_PATH' in key]

for table in tables_to_optimize:
    optimize_delta(table)

