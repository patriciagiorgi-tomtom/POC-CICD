variable "databricks_host" {
  description = "Databricks workspace URL"
}

variable "databricks_token" {
  description = "Databricks API token"
}

variable "pool_name" {
  description = "Name of the instance pool"
  default     = "example-patricia-pool"
}

variable "node_type_id" {
  description = "Databricks node type ID"
  default     = "Standard_DS3_v2"
}

terraform {
  required_providers {
    databricks = {
      source  = "databricks/databricks"
      version = "1.0.0"
    }
  }
}