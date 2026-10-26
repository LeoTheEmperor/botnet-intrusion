#!/usr/bin/env bash
# First-boot setup for the Raspberry Pi. Run it from anywhere inside the cloned repository:
#     bash scripts/pi/setup_pi.sh                    # install packages + run the benchmark
#     bash scripts/pi/setup_pi.sh --install-services # ... and install the systemd units for the current user
set -euo pipefail
cd "$(dirname "$0")/../.."
REPO="$(pwd)"

echo "== 1/4 installing packages (numpy, git, gpiozero) =="
command -v apt-get >/dev/null || { echo "apt-get not found: this script is for Raspberry Pi OS / Debian"; exit 1; }
sudo apt-get update -y
sudo apt-get install -y python3-numpy git python3-gpiozero

echo; echo "== 2/4 network facts you will need =="
echo "interfaces:"; ip -br link | sed 's/^/  /'
echo "this Pi's address(es): $(hostname -I)"
echo "(Wi-Fi power saving can add latency jitter; to switch it off for this boot: sudo iw dev wlan0 set power_save off)"

echo; echo "== 3/4 benchmark: how many packets per second can this Pi sustain? =="
if [ -f models/live_model.npz ]; then python3 src/benchmark.py --model models/live_model.npz; else python3 src/benchmark.py; fi

echo; echo "== 4/4 services =="
if [ "${1:-}" = "--install-services" ]; then
  U="$(id -un)"
  for f in botnet-detector led-indicator; do
    sed -e "s|YOUR_USER|$U|g" -e "s|/home/$U/botnet-intrusion|$REPO|g" "scripts/pi/$f.service" | sudo tee "/etc/systemd/system/$f.service" >/dev/null
  done
  sudo systemctl daemon-reload
  if [ -f models/live_model.npz ]; then
    sudo systemctl enable --now botnet-detector led-indicator
    echo "services installed and started (journalctl -u botnet-detector -f)"
  else
    echo "services installed but NOT started: copy models/live_model.npz first, then: sudo systemctl enable --now botnet-detector led-indicator"
  fi
else
  echo "skipped (add --install-services to install the systemd units)"
fi
echo; echo "done. Next: docs/PI_SETUP.md section 4 (record training data)."
