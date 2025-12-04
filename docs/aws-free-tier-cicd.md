# AWS Free-Tier CI/CD Pipeline Setup

This guide explains how to deploy this repository automatically to AWS using only free-tier friendly services.

## 1. Architecture Overview

- **GitHub Actions** builds the container image and orchestrates deployment.
- **Amazon ECR** stores images (first 500 MB/month free).
- **Amazon EC2 t2.micro** runs the container (750 hours/month free for first 12 months).
- **AWS Systems Manager (SSM)** runs remote commands to refresh the container (3 managed instances free).
- **Amazon CloudWatch** provides basic monitoring at no additional cost.

Pipeline flow: push to `main` → GitHub Actions builds & pushes image → GitHub Actions triggers SSM → EC2 pulls latest image → container restarts.

## 2. Prerequisites

1. AWS Account within free-tier limits.
2. GitHub repository containing this project.
3. AWS CLI installed locally (for one-time setup) with credentials that can administer IAM, EC2, ECR, and SSM.
4. Docker installed locally to validate builds (optional).

## 3. Create AWS Resources

### 3.1 Terraform one-click provisioning (recommended)

All infrastructure required for the pipeline can be stood up with Terraform in `infra/terraform/`:

```bash
cd infra/terraform
terraform init
terraform apply \
  -var="region=ap-south-1" \
  -var="vpc_id=vpc-xxxxxxxx" \
  -var="subnet_id=subnet-xxxxxxxx" \
  -var="key_name=your-keypair" # optional, use "" for none
```

Key outputs after apply:

| Output | Use |
|--------|-----|
| `ecr_repository_url` | Copy into the `ECR_REPOSITORY` GitHub secret. |
| `instance_id` | Copy into `TARGET_INSTANCE_ID`. |
| `instance_public_ip` | Quick access URL for manual testing. |

Variables are defined in `variables.tf`; adjust defaults (container ports, instance type, tags) as needed. Terraform also bakes the `/opt/app/deploy.sh` script onto the instance via user data templates under `infra/terraform/templates/`.

If you prefer console setup instead, follow sections 3.2–3.4.

### 3.2 Elastic Container Registry (ECR)

```bash
aws ecr create-repository \
  --repository-name inai-backend \
  --image-scanning-configuration scanOnPush=true
```
```bash
aws ecr create-repository \
  --repository-name inai-backend \
  --image-scanning-configuration scanOnPush=true
```
Record the repository URI (e.g. `123456789012.dkr.ecr.ap-south-1.amazonaws.com/inai-backend`).

### 3.3 IAM for GitHub Actions (OIDC)

1. **OIDC Provider**: IAM → Identity providers → Add provider.
   - Provider URL: `https://token.actions.githubusercontent.com`
   - Audience: `sts.amazonaws.com`
2. **Role `GitHubDeployRole`**: Create IAM role with trusted entity **Web identity** referencing the provider.
   - Restrict `sub` to the repo/branch:
     ```json
     "Condition": {
       "StringEquals": {
         "token.actions.githubusercontent.com:aud": "sts.amazonaws.com",
         "token.actions.githubusercontent.com:sub": "repo:OWNER/REPO:ref:refs/heads/main"
       }
     }
     ```
3. Attach inline policy:
   ```json
   {
     "Version": "2012-10-17",
     "Statement": [
       {
         "Sid": "FrontendArtifactBucket",
         "Effect": "Allow",
         "Action": [
           "s3:PutObject",
           "s3:GetObject",
           "s3:DeleteObject"
         ],
         "Resource": "arn:aws:s3:::<FRONTEND_S3_BUCKET>/*"
       },
       {
         "Sid": "FrontendArtifactBucketList",
         "Effect": "Allow",
         "Action": "s3:ListBucket",
         "Resource": "arn:aws:s3:::<FRONTEND_S3_BUCKET>"
       },
       {
         "Sid": "ECRManagement",
         "Effect": "Allow",
         "Action": [
           "ecr:BatchGetImage",
           "ecr:BatchCheckLayerAvailability",
           "ecr:CompleteLayerUpload",
           "ecr:GetDownloadUrlForLayer",
           "ecr:InitiateLayerUpload",
           "ecr:PutImage",
           "ecr:UploadLayerPart",
           "ecr:DescribeRepositories"
         ],
         "Resource": "arn:aws:ecr:ap-south-1:<ACCOUNT_ID>:repository/inai-backend"
       },
       {
         "Sid": "ECRGetToken",
         "Effect": "Allow",
         "Action": "ecr:GetAuthorizationToken",
         "Resource": "*"
       },
       {
         "Sid": "SSMDeployment",
         "Effect": "Allow",
         "Action": [
           "ssm:SendCommand",
           "ssm:GetCommandInvocation",
           "ssm:ListCommandInvocations",
           "ssm:ListCommands"
         ],
         "Resource": [
           "arn:aws:ssm:ap-south-1:*:document/${DEPLOYMENT_SSM_DOCUMENT}",
           "arn:aws:ec2:ap-south-1:<ACCOUNT_ID>:instance/*"
         ],
         "Condition": {
           "StringLike": {
             "ssm:resourceTag/Name": "ci-managed"
           }
         }
       },
       {
         "Sid": "DescribeInstances",
         "Effect": "Allow",
         "Action": ["ec2:DescribeInstances"],
         "Resource": "*"
       }
     ]
   }
   ```
4. Save the role ARN for GitHub secrets.

### 3.4 EC2 Instance

1. Launch Amazon Linux 2 **t2.micro** in default VPC.
2. Attach IAM role `EC2ContainerRole` with permissions:
   ```json
   {
     "Version": "2012-10-17",
     "Statement": [
       {
         "Sid": "PullFromEcr",
         "Effect": "Allow",
         "Action": [
           "ecr:GetAuthorizationToken",
           "ecr:BatchCheckLayerAvailability",
           "ecr:GetDownloadUrlForLayer",
           "ecr:BatchGetImage"
         ],
         "Resource": "*"
       },
       {
         "Sid": "SSMManaged",
         "Effect": "Allow",
         "Action": [
           "ssm:UpdateInstanceInformation",
           "ssmmessages:*",
           "ec2messages:*"
         ],
         "Resource": "*"
       }
     ]
   }
   ```
   > Alternatively attach AWS managed policy `AmazonEC2RoleforSSM`.
3. Add security group rules: inbound HTTP (80) as needed; outbound all.
4. Tag instance with `Name=ci-managed` to satisfy SSM condition.

### 3.5 Bootstrap Deployment Script

Run once via SSM or SSH:

```bash
sudo mkdir -p /opt/app
sudo chown ec2-user:ec2-user /opt/app
cat <<'EOF' > /opt/app/deploy.sh
#!/usr/bin/env bash
$(cat ../scripts/deploy.sh)
EOF
chmod +x /opt/app/deploy.sh
```

Adjust the script values to match your region/ECR URI before uploading.

## 4. Configure GitHub Secrets

In repository settings → *Secrets and variables* → *Actions*, add:

| Secret | Description |
|--------|-------------|
| `AWS_REGION` | AWS region (e.g. `ap-south-1`). |
| `ECR_REPOSITORY` | Full ECR URI (`<acct>.dkr.ecr.<region>.amazonaws.com/inai-backend`). Terraform output `ecr_repository_url` provides this. |
| `AWS_ROLE_TO_ASSUME` | ARN of `GitHubDeployRole`. |
| `DEPLOYMENT_SSM_DOCUMENT` | `AWS-RunShellScript` or custom document ARN. |
| `TARGET_INSTANCE_ID` | EC2 instance ID. Terraform output `instance_id` provides this. |

> Run `terraform output` after apply to copy values directly into the secrets.

## 5. GitHub Actions Workflow

Defined in `.github/workflows/deploy.yml`. It:
1. Checks out code.
2. Assumes the OIDC role.
3. Logs in to ECR.
4. Builds and tags Docker image (`SHA` + `latest`).
5. Pushes both tags to ECR.
6. Triggers SSM Run Command to run `/opt/app/deploy.sh <image_sha>` on the instance.

## 6. Testing the Pipeline

1. Commit and push to `main`.
2. Observe GitHub Actions run under **Actions** tab.
3. In AWS Console → Systems Manager → Run Command → View results to confirm execution.
4. Access the EC2 public IP (port 80) to verify the application.

## 7. Cost Governance Tips

- Monitor AWS Billing dashboard for free-tier usage.
- Stop the EC2 instance when idle to remain within 750 hours.
- Periodically prune ECR images (`ecr:BatchDeleteImage`) to stay under 500 MB.
- Consider using `t4g.micro` (Graviton) if offered within free tier in your region.

## 8. Troubleshooting

| Symptom | Action |
|---------|--------|
| Workflow fails at credentials step | Confirm OIDC trust policy matches repo path and branch. |
| `AccessDeniedException` on SSM | Ensure instance tagged `ci-managed` and document ARN matches. |
| Docker permission denied on EC2 | Reboot after installing Docker or run `sudo usermod -a -G docker ec2-user`. |
| Container not reachable | Check security group inbound rules and container logs (`docker logs inai-backend`). |

Once configured, deployments occur automatically on every push to `main`, fully within AWS free-tier allowances.
