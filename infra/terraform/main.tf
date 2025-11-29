terraform {
  required_version = ">= 1.5.0"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = var.region
}

data "aws_ssm_parameter" "amzn2" {
  name = "/aws/service/ami-amazon-linux-latest/amzn2-ami-hvm-x86_64-gp2"
}

resource "aws_ecr_repository" "app" {
  name = var.ecr_repository_name

  image_scanning_configuration {
    scan_on_push = true
  }

  encryption_configuration {
    encryption_type = "AES256"
  }

  tags = var.common_tags
}

resource "aws_iam_role" "ec2_container_role" {
  name = var.iam_role_name

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Principal = {
          Service = "ec2.amazonaws.com"
        }
        Action = "sts:AssumeRole"
      }
    ]
  })

  tags = var.common_tags
}

resource "aws_iam_role_policy_attachment" "ssm" {
  role       = aws_iam_role.ec2_container_role.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonEC2RoleforSSM"
}

resource "aws_iam_role_policy" "ecr_pull" {
  name = "AllowECRPull"
  role = aws_iam_role.ec2_container_role.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect   = "Allow"
        Action   = [
          "ecr:GetAuthorizationToken",
          "ecr:BatchCheckLayerAvailability",
          "ecr:GetDownloadUrlForLayer",
          "ecr:BatchGetImage"
        ]
        Resource = "*"
      }
    ]
  })
}

resource "aws_iam_instance_profile" "this" {
  name = "${var.iam_role_name}-profile"
  role = aws_iam_role.ec2_container_role.name
}

resource "aws_security_group" "ci_host" {
  name        = "ci-cd-host-sg"
  description = "Allow HTTP ingress for CI/CD host"
  vpc_id      = var.vpc_id

  ingress {
    description = "HTTP inbound"
    from_port   = var.host_port
    to_port     = var.host_port
    protocol    = "tcp"
    cidr_blocks = [var.allowed_cidr]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = var.common_tags
}

locals {
  ecr_repository_url    = aws_ecr_repository.app.repository_url
  deploy_script         = templatefile("${path.module}/templates/deploy.sh.tpl", {
    aws_region     = var.region
    ecr_uri        = local.ecr_repository_url
    container_name = var.container_name
    container_port = var.container_port
    host_port      = var.host_port
  })
  deploy_script_base64 = base64encode(local.deploy_script)
  user_data            = templatefile("${path.module}/templates/user_data.sh.tpl", {
    deploy_script_base64 = local.deploy_script_base64
    host_port            = var.host_port
  })
  instance_tags = merge(var.common_tags, {
    Name = "ci-managed"
  })
}

resource "aws_instance" "ci_host" {
  ami                         = data.aws_ssm_parameter.amzn2.value
  instance_type               = var.instance_type
  subnet_id                   = var.subnet_id
  vpc_security_group_ids      = [aws_security_group.ci_host.id]
  iam_instance_profile        = aws_iam_instance_profile.this.name
  associate_public_ip_address = true
  key_name                    = var.key_name
  user_data                   = local.user_data

  tags = local.instance_tags
}

``
