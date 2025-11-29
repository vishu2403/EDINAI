variable "region" {
  description = "AWS region to deploy resources in"
  type        = string
}

variable "vpc_id" {
  description = "VPC ID where the CI/CD host should live"
  type        = string
}

variable "subnet_id" {
  description = "Subnet ID with internet access (public subnet recommended)"
  type        = string
}

variable "key_name" {
  description = "Optional EC2 key pair name for SSH access"
  type        = string
  default     = ""
}

variable "allowed_cidr" {
  description = "CIDR block allowed to reach the host port"
  type        = string
  default     = "0.0.0.0/0"
}

variable "instance_type" {
  description = "EC2 instance type for the deployment host"
  type        = string
  default     = "t2.micro"
}

variable "container_port" {
  description = "Internal container port exposed by the workload"
  type        = number
  default     = 8000
}

variable "host_port" {
  description = "Host port to expose the container on"
  type        = number
  default     = 80
}

variable "container_name" {
  description = "Name to give the running container"
  type        = string
  default     = "inai-backend"
}

variable "ecr_repository_name" {
  description = "Name for the ECR repository"
  type        = string
  default     = "inai-backend"
}

variable "iam_role_name" {
  description = "Name for the IAM role attached to the EC2 instance"
  type        = string
  default     = "EC2ContainerRole"
}

variable "common_tags" {
  description = "Map of tags to apply to all resources"
  type        = map(string)
  default     = {}
}
