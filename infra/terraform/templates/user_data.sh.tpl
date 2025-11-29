#!/bin/bash
set -euo pipefail

# Write deployment script to /opt/app
mkdir -p /opt/app
cat <<'EOF' > /opt/app/deploy.sh
${deploy_script_base64}
EOF

# Decode base64 payload
base64 -d /opt/app/deploy.sh > /opt/app/deploy.sh.dec
mv /opt/app/deploy.sh.dec /opt/app/deploy.sh
chmod +x /opt/app/deploy.sh
chown ec2-user:ec2-user /opt/app/deploy.sh

# Ensure docker group exists for ec2-user after reboot
usermod -aG docker ec2-user || true

# Open firewall for host port if using firewalld (Amazon Linux doesn't by default)
if command -v firewall-cmd >/dev/null 2>&1; then
  firewall-cmd --permanent --add-port=${host_port}/tcp || true
  firewall-cmd --reload || true
fi

# Run initial deployment with latest tag
/opt/app/deploy.sh latest
