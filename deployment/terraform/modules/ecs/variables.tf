variable "environment" {
  description = "Deployment environment (dev, staging, production)"
  type        = string
}

variable "cluster_name" {
  description = "Name of the ECS cluster"
  type        = string
}

variable "region" {
  description = "AWS region"
  type        = string
}

variable "image_uri" {
  description = "Docker image URI for the application container"
  type        = string
}

variable "cpu" {
  description = "Fargate task CPU units (1024 = 1 vCPU)"
  type        = number
  default     = 512
}

variable "memory" {
  description = "Fargate task memory in MiB"
  type        = number
  default     = 1024
}

variable "desired_count" {
  description = "Desired number of running tasks"
  type        = number
  default     = 2
}

variable "container_port" {
  description = "Port the container listens on"
  type        = number
  default     = 8000
}

variable "vpc_id" {
  description = "VPC ID for the ECS service networking"
  type        = string
}

variable "subnet_ids" {
  description = "Subnet IDs for the ECS service and ALB"
  type        = list(string)
}
