# Raspberry Pi Zero 2 W: setup and demo runbook

> Nothing in this guide has been run on a real Pi yet; the live path was tested on a laptop (loopback capture, replayed traffic).
> Everything below is the plan to follow once the board arrives. Use only networks and devices you own.

## Before the Pi arrives (do these now)

1. **Rehearse on the laptop.** `pip install numpy scapy`, then `python src/demo_replay.py`. It generates synthetic traffic, trains a model and replays unseen traffic
   through the detector with the dashboard (opens http://localhost:8080, about a minute). Check the page on your screen and on a phone (`--lan`), and note anything to change.
   It is synthetic traffic: say so when you show it. This is also your **fallback demo** if anything goes wrong on the day.
2. **Pre-flash the microSD card** (section 1) with the Wi-Fi of the network you will actually demo on, so on Monday it is plug in, boot, SSH.
3. **Parts:** buy the LEDs, resistors, breadboard and jumper wires locally if you have not got them. Test the logic now with `python src/led_indicator.py --simulate`.
4. **Demo network:** your own router or a phone hotspot, with client isolation off. Confirm your laptop can reach another device on it.
5. **Laptop side:** run `python src/traffic_sim.py syn --target 127.0.0.2 --duration 3 --i-own-this-network` to confirm the generator works on your operating system.
6. Run the full notebook once, and commit the repository.

## 0. Parts

Pi Zero 2 W (WH, header pre-soldered), 5V 2.5A micro-USB supply with a short cable, 2 microSD cards (16-32 GB, A1), card reader,
3 LEDs (green, yellow, red) + 3 resistors of 220-330 ohm + breadboard + jumper wires, optional power bank.
A second machine (your laptop) to generate attack traffic, and a network you own: your own router, or a phone hotspot with client isolation off.

## 1. Flash the OS (on the laptop)

1. Install **Raspberry Pi Imager**. Device: *Raspberry Pi Zero 2 W*; OS: *Raspberry Pi OS Lite (32-bit)* (lighter on a 512 MB board; 64-bit also works).
2. Open the settings (Edit Settings) and fill in: hostname (e.g. `botnetpi`), username + password, **Wi-Fi name, password and country**, enable **SSH**.
3. Write the card, put it in the Pi, power it, and wait 2-5 minutes for the first boot.

## 2. First login and install

```bash
ssh <user>@botnetpi.local          # if .local does not resolve, find the IP in your router / hotspot client list
sudo apt update
sudo apt install -y python3-numpy git python3-gpiozero
git clone https://github.com/LeoTheEmperor/botnet-intrusion.git && cd botnet-intrusion
```

**Shortcut:** `bash scripts/pi/setup_pi.sh` does the three `apt` steps, prints the interface name and address, and runs the benchmark
(add `--install-services` later to install the systemd units for your user).

Do not install pandas on the Pi: it is only needed for training, which we do on the laptop. (No repo access? `scp` the zip over instead.)
Useful: `hostname -I` (the Pi's IP), `ip -br link` (interface name, normally `wlan0`).

## 3. Benchmark first (do this before anything else)

```bash
python3 src/benchmark.py
```

It prints microseconds per packet for the parser and the feature extractor and a rule-of-thumb maximum flood rate. On the development laptop the full
detector sustains about 12,000 packets/s; the Pi Zero 2 W will be several times slower. **Keep every test flood below the printed rule of thumb**
(`traffic_sim.py --rate N`), otherwise the Pi falls behind and drops packets. Optional: `sudo iw dev wlan0 set power_save off` for steadier Wi-Fi latency.

## 4. Record training data: continuous sessions, labelled by markers

The long-window features remember history, so record *continuous* sessions, not one file per class. The generator sends a tiny UDP marker at the start of each
segment and `collect` labels the capture from those markers, so no clock synchronisation is needed.

On the Pi (leave it running):
```bash
sudo python3 src/live_detector.py collect --iface wlan0 --markers benign,syn_flood,udp_flood,port_scan \
     --trim-edges 1 --warmup 100 --out data/live/session1.csv
```
On the laptop, **after** collect is running (start it within a few seconds, any delay is simply ignored because rows before the first marker are dropped):
```bash
python3 src/traffic_sim.py session --target <PI_IP> --i-own-this-network \
     --plan "benign:120,syn_flood:40,benign:40,udp_flood:40,benign:40,port_scan:40,benign:60"
```
Stop collect with Ctrl+C when the session ends. Record **at least two sessions** (e.g. `session1.csv`, `session2.csv`, with the attack order changed).
Keep your own SSH session quiet while recording (its traffic is part of the capture).

## 5. Train on the laptop and copy the model back

```bash
scp <user>@botnetpi.local:botnet-intrusion/data/live/*.csv data/live/
python3 src/train_live_model.py --data data/live --out models/live_model.npz --final
scp models/live_model.npz <user>@botnetpi.local:botnet-intrusion/models/
```
The trainer prints a time-ordered hold-out accuracy per class. For an honest test, record one more session *after* training and never train on it.

## 6. Run the demo

Pi, terminal 1:
```bash
sudo python3 src/live_detector.py detect --iface wlan0 --model models/live_model.npz --dashboard 8080 --dashboard-host 0.0.0.0
```
Open `http://<PI_IP>:8080` on the laptop or a phone (same network). Pi, terminal 2 (optional LEDs):
```bash
python3 src/led_indicator.py
```
Laptop, one attack at a time (rates below the benchmark's rule of thumb):
```bash
python3 src/traffic_sim.py syn  --target <PI_IP> --rate 300 --duration 20 --i-own-this-network
python3 src/traffic_sim.py udp  --target <PI_IP> --rate 300 --duration 20 --i-own-this-network
python3 src/traffic_sim.py scan --target <PI_IP> --rate 300 --duration 20 --i-own-this-network
```
The detector ignores the dashboard's own HTTP traffic automatically.

## 7. LED wiring (BCM numbering)

```
GPIO17 (pin 11) --[220-330 ohm]--(+) GREEN  (-)--+
GPIO27 (pin 13) --[220-330 ohm]--(+) YELLOW (-)--+--> GND (pin 14)
GPIO22 (pin 15) --[220-330 ohm]--(+) RED    (-)--+
```
Long leg of an LED = anode (+, towards the resistor); short leg = cathode (-, towards GND).
Green steady = normal; yellow steady = suspicious; **red blinking = alert**; yellow blinking = waiting or no data; all off = replay finished; red/yellow alternating = detector unreachable.
No hardware: `python3 src/led_indicator.py --simulate`.

## 8. Start automatically at boot (optional)

Edit `YOUR_USER` in `scripts/pi/*.service`, then:
```bash
sudo cp scripts/pi/*.service /etc/systemd/system/ && sudo systemctl daemon-reload
sudo systemctl enable --now botnet-detector led-indicator
journalctl -u botnet-detector -f        # live log
```
The unit uses `CAP_NET_RAW` instead of running as root; if raw capture fails, set `User=root` in the unit.

## 9. Fallback if live traffic misbehaves on demo day

See [`DEMO_OPTIONS.md`](DEMO_OPTIONS.md): a ladder of four fallbacks (dataset replay, laptop live, synthetic replay, recording), what to say at each, and a 30-second triage table.
The best one is `python3 src/demo_nbaiot.py`: it needs only numpy, so it also runs on the Pi, with no network capture at all.

## 10. Troubleshooting

| Symptom | Likely cause / fix |
|---|---|
| `Operation not permitted` on capture | needs `sudo` (raw socket) |
| `No such device` | wrong interface name; check `ip -br link` |
| no packets, nothing on the dashboard | Wi-Fi client isolation (use your own router or hotspot), or the Pi is idle: ping it from the laptop |
| dashboard unreachable from the laptop | forgot `--dashboard-host 0.0.0.0`, different networks, or a firewall |
| everything labelled benign | model trained on too little or different traffic; record more sessions, same network, same rates |
| detector falls behind, windows lag | flood rate above the benchmark's limit: lower `--rate`, or `detect --every 5` |
| random reboots | weak power supply or long cable |
| LEDs do nothing | wrong pins or polarity; test with `python3 src/led_indicator.py --simulate` first |

## Demo-day checklist

Charged power bank or mains socket; both devices on the same network; `hostname -I` noted; model copied to `models/`; dashboard opened on a second device;
LEDs wired and tested; benchmark rule of thumb known; fallback replay and a laptop copy of the pipeline ready; the report and slides open.
