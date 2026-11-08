#!/usr/bin/env bash
# Install the official GitHub CLI to ~/.local/bin (no sudo), checksum-verified.
#   bash infra/install_gh.sh && gh auth login
set -euo pipefail
V=$(curl -fsSL https://api.github.com/repos/cli/cli/releases/latest |
  python3 -c 'import sys, json; print(json.load(sys.stdin)["tag_name"][1:])')
T="gh_${V}_linux_amd64"
cd "$(mktemp -d)"
curl -fsSLO "https://github.com/cli/cli/releases/download/v${V}/${T}.tar.gz"
curl -fsSL "https://github.com/cli/cli/releases/download/v${V}/gh_${V}_checksums.txt" |
  grep " ${T}.tar.gz$" | sha256sum -c -
tar xzf "${T}.tar.gz"
mkdir -p "$HOME/.local/bin"
install -m755 "${T}/bin/gh" "$HOME/.local/bin/gh"
"$HOME/.local/bin/gh" --version | head -1
