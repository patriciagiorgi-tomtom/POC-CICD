provider "databricks" {
  host  = var.databricks_host
  token = var.databricks_token
}

resource "databricks_instance_pool" "DSv2" {
  instance_pool_name = var.pool_DSv2_name
  min_idle_instances = 0
  max_capacity       = 100
  node_type_id       = var.pool_DSv2_node_type_id
  idle_instance_autotermination_minutes = 15
}

resource "databricks_instance_pool" "F8s" {
  instance_pool_name = var.pool_F8s_v2_name
  min_idle_instances = 0
  max_capacity       = 100
  node_type_id       = var.pool_F8s_node_type_id
  idle_instance_autotermination_minutes = 15
}
resource "databricks_instance_pool" "E8d" {
  instance_pool_name = var.pool_E8d_v4_name
  min_idle_instances = 0
  max_capacity       = 100
  node_type_id       = var.pool_E8d_v4_node_type_id
  idle_instance_autotermination_minutes = 15
}
resource "databricks_instance_pool" "NC4as_T4" {
  instance_pool_name = var.pool_NC4as_T4_pool_name
  min_idle_instances = 0
  max_capacity       = 100
  node_type_id       = var.pool_NC4as_T4_node_type_id
  idle_instance_autotermination_minutes = 15
}

