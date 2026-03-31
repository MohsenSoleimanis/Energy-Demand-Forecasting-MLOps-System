output "eks_cluster_endpoint" {
  description = "EKS cluster endpoint"
  value       = module.eks.cluster_endpoint
}

output "rds_endpoint" {
  description = "RDS endpoint for MLflow"
  value       = aws_db_instance.mlflow.endpoint
}

output "lakehouse_bucket" {
  description = "S3 bucket for data lakehouse"
  value       = aws_s3_bucket.lakehouse.id
}

locals {
  tags = {
    Project     = var.project_name
    Environment = var.environment
    ManagedBy   = "terraform"
  }
}
