output "ecr_repository_url" {
  description = "Full URI of the ECR repository used by the pipeline"
  value       = aws_ecr_repository.app.repository_url
}

output "instance_id" {
  description = "EC2 instance ID for the CI/CD host"
  value       = aws_instance.ci_host.id
}

output "instance_public_ip" {
  description = "Public IPv4 address of the CI/CD host"
  value       = aws_instance.ci_host.public_ip
}
