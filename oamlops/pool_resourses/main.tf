provider "databricks" {
  host  = var.databricks_host
  token = var.databricks_token
}

resource "databricks_instance_pool" "this" {
  instance_pool_name = var.pool_name
  min_idle_instances = 0
  max_capacity       = 100
  node_type_id       = var.node_type_id
  idle_instance_autotermination_minutes = 15
}
