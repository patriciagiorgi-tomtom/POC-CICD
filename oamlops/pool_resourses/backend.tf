 terraform {    
    backend "azurerm" {
    resource_group_name   = "reksio-databricks-external-storage-rg"
    storage_account_name  = "reksiodbxstorageaccount"
    container_name        = "mlops-terraform-cicd"
    key                   = "stateActions.tfstate"
  }
 }
