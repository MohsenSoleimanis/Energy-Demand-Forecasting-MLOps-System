resource "aws_s3_bucket" "lakehouse" {
  bucket = "${var.project_name}-lakehouse-${var.environment}"
  tags   = local.tags
}

resource "aws_s3_bucket" "mlflow_artifacts" {
  bucket = "${var.project_name}-mlflow-artifacts-${var.environment}"
  tags   = local.tags
}

resource "aws_s3_bucket" "dvc_storage" {
  bucket = "${var.project_name}-dvc-storage-${var.environment}"
  tags   = local.tags
}

resource "aws_s3_bucket_versioning" "lakehouse" {
  bucket = aws_s3_bucket.lakehouse.id
  versioning_configuration { status = "Enabled" }
}

resource "aws_s3_bucket_lifecycle_configuration" "lakehouse" {
  bucket = aws_s3_bucket.lakehouse.id

  rule {
    id     = "archive-old-bronze"
    status = "Enabled"
    filter { prefix = "bronze/" }
    transition {
      days          = 90
      storage_class = "STANDARD_IA"
    }
    transition {
      days          = 365
      storage_class = "GLACIER"
    }
  }
}
