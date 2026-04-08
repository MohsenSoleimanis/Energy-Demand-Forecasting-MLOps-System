output "api_endpoint" {
  description = "URL of the Energy Forecast API load balancer"
  value       = module.ecs.alb_dns_name
}

output "mlflow_uri" {
  description = "MLflow tracking server URI"
  value       = "http://${module.ecs.alb_dns_name}:5000"
}

output "ecs_cluster_name" {
  description = "Name of the ECS cluster"
  value       = module.ecs.cluster_name
}

output "ecs_service_name" {
  description = "Name of the ECS service"
  value       = module.ecs.service_name
}
