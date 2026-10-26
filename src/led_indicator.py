#!/usr/bin/env python3
"""
Traffic-light indicator for the Raspberry Pi, driven by the detector's dashboard API.

  green  steady   = normal traffic            yellow steady   = suspicious (attack class, below alert threshold)
  red    blinking = ALERT                     yellow blinking = waiting for traffic / no data
  all off         = replay finished           red/yellow alternating = detector not reachable

Start the detector with --dashboard 8080 (any --dashboard-host), then run this script on the same Pi:
    python3 src/led_indicator.py                  # BCM pins 17 / 27 / 22
    python3 src/led_indicator.py --simulate       # no hardware: print the LED pattern in the terminal

Wiring (BCM numbering; every LED through a 220-330 ohm resistor to GND):
    green  -> GPIO17 (physical pin 11)     yellow -> GPIO27 (pin 13)     red -> GPIO22 (pin 15)     GND -> pin 14
Needs gpiozero on the Pi:  sudo apt install -y python3-gpiozero
"""
import argparse
import json
import time
import urllib.request


def fetch(url, timeout=1.0):
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return json.loads(r.read().decode())
    except Exception:
        return None


def decide(s):
    """Map a dashboard snapshot to one of: offline, waiting, nodata, finished, alert, suspicious, ok."""
    if s is None:
        return "offline"
    h = s.get("history") or []
    if not h:
        return "waiting"
    if s.get("finished"):
        return "finished"
    if s["server_time"] - (s.get("last_push") or 0) > max(5, 3 * s.get("window_s", 1)):
        return "nodata"
    cur = h[-1]
    if cur["alert"]:
        return "alert"
    return "ok" if cur["label"] == s.get("benign_label", "benign") else "suspicious"


def blink(now, hz):
    return int(now * hz * 2) % 2 == 0


def pattern(state, now):
    """(green, yellow, red) for a state at time `now`."""
    return {"ok": (1, 0, 0), "suspicious": (0, 1, 0), "alert": (0, 0, blink(now, 4)),
            "waiting": (0, blink(now, 1), 0), "nodata": (0, blink(now, 2), 0), "finished": (0, 0, 0),
            "offline": (0, blink(now, 1), not blink(now, 1))}[state]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://127.0.0.1:8080/api/state")
    ap.add_argument("--pins", default="17,27,22", help="BCM pins for green,yellow,red")
    ap.add_argument("--interval", type=float, default=0.5, help="seconds between polls")
    ap.add_argument("--simulate", action="store_true", help="print the pattern instead of driving GPIO")
    ap.add_argument("--duration", type=float, default=None, help="stop after N seconds (testing)")
    a = ap.parse_args()

    leds = None
    if not a.simulate:
        from gpiozero import LED
        leds = [LED(int(p)) for p in a.pins.split(",")]
    state, last_poll, last_printed, t_end = "offline", 0.0, None, (time.time() + a.duration if a.duration else None)
    try:
        while t_end is None or time.time() < t_end:
            now = time.time()
            if now - last_poll >= a.interval:
                state, last_poll = decide(fetch(a.url)), now
            g, y, r = (int(bool(v)) for v in pattern(state, now))
            if leds:
                for led, v in zip(leds, (g, y, r)):
                    led.on() if v else led.off()
            elif state != last_printed:
                print(f"[{time.strftime('%H:%M:%S')}] state={state:<10s} green={'ON' if state == 'ok' else '--'}  "
                      f"yellow={'ON' if state == 'suspicious' else ('blink' if state in ('waiting', 'nodata', 'offline') else '--')}  "
                      f"red={'blink' if state == 'alert' else ('alt' if state == 'offline' else '--')}", flush=True)
                last_printed = state
            time.sleep(0.05)
    except KeyboardInterrupt:
        pass
    finally:
        if leds:
            for led in leds:
                led.off()


if __name__ == "__main__":
    main()
