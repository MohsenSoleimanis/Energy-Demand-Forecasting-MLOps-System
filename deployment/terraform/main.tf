terraform {
  required_version = ">= 1.5.0"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }

  backend "s3" {
    bucket         = "energy-forecast-terraform-state"
    key            = "infrastructure/terraform.tfstate"
    region         = "us-east-1"
    dynamodb_table = "terraform-lock"
    encrypt        = true
  }
}

provider "aws" {
  region = var.region

  default_tags {
    tags = {
      Project     = "energy-demand-forecasting"
      Environment = var.environment
      ManagedBy   = "terraform"
    }
  }
}

module "ecs" {
  source = "./modules/ecs"

  environment    = var.environment
  cluster_name   = var.cluster_name
  region         = var.region
  image_uri      = var.image_uri
  cpu            = var.cpu
  memory         = var.memory
  desired_count  = var.desired_count
  vpc_id         = var.vpc_id
  subnet_ids     = var.subnet_ids
  container_port = 8000
}
