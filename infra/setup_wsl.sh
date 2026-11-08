#!/usr/bin/env bash
# One-time system setup inside WSL Ubuntu 24.04. Needs sudo; run it yourself:
#   bash infra/setup_wsl.sh
# Installs: Docker Engine (official apt repo), NVIDIA Container Toolkit, build essentials.
# The GPU driver comes from Windows; do NOT install a Linux NVIDIA driver in WSL.
set -euo pipefail

if [[ $EUID -eq 0 ]]; then echo "run as your normal user (script calls sudo)"; exit 1; fi

sudo apt-get update
sudo apt-get install -y ca-certificates curl gnupg build-essential git git-lfs jq pigz

# --- Docker Engine -----------------------------------------------------------
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor --yes -o /etc/apt/keyrings/docker.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo "$VERSION_CODENAME") stable" |
  sudo tee /etc/apt/sources.list.d/docker.list >/dev/null
sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin
sudo usermod -aG docker "$USER"

# --- NVIDIA Container Toolkit (GPU inside containers; used for optional containerised vLLM) --
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey |
  sudo gpg --dearmor --yes -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list |
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' |
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list >/dev/null
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker

# cgroup v2 is required by rbcore.hwmon.docker_stats (fast per-container CPU/RAM/OOM reads).
stat -fc %T /sys/fs/cgroup | grep -q cgroup2fs || echo "WARNING: cgroup v2 not active"

sudo systemctl enable --now docker
sudo systemctl restart docker

echo
echo "Done. Open a NEW WSL shell (docker group), then run: bash infra/bootstrap_user.sh"
