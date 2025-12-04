output "ecr_repository_url" {
  description = "Full URI of the ECR repository used by the pipeline"
  value       = aws_ecr_repository.app.repository_url
}

output "instance_id" {
  description = "EC2 instance ID for the CI/CD host"
  value       = aws_instance.ci_host.id
}

output "instance_public_ip" {
  description = "Elastic IP address assigned to the CI/CD host"
  value       = aws_eip.elastic_ip.public_ip
}

output "elastic_ip_allocation_id" {
  description = "Allocation ID of the Elastic IP (for AWS console reference)"
  value       = aws_eip.elastic_ip.id
}

# ALB Outputs
output "alb_dns_name" {
  description = "The DNS name of the load balancer"
  value       = aws_lb.app_alb.dns_name
}

output "alb_zone_id" {
  description = "The zone ID of the load balancer to assist with creating DNS records"
  value       = aws_lb.app_alb.zone_id
}

output "alb_arn" {
  description = "The ARN of the load balancer"
  value       = aws_lb.app_alb.arn
}

# Auto Scaling Group Outputs
output "asg_name" {
  description = "The name of the Auto Scaling Group"
  value       = aws_autoscaling_group.app_asg.name
}

output "asg_arn" {
  description = "The ARN of the Auto Scaling Group"
  value       = aws_autoscaling_group.app_asg.arn
}

# Target Group Outputs
output "target_group_arn" {
  description = "The ARN of the target group"
  value       = aws_lb_target_group.app_tg.arn
}

output "target_group_name" {
  description = "The name of the target group"
  value       = aws_lb_target_group.app_tg.name
}

# CloudWatch Dashboard
output "cloudwatch_dashboard_url" {
  description = "URL of the CloudWatch dashboard"
  value       = "https://${var.region}.console.aws.amazon.com/cloudwatch/home?region=${var.region}#dashboards:name=${var.environment}-app-dashboard"
}
