#!/usr/bin/env python3
# tools/alarm.py — runs as its own process
# Usage: python alarm.py "set an alarm for 7:30 a.m."
import os
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Suppress ALSA noise before any audio import
_dn = os.open(os.devnull, os.O_WRONLY)
_se = os.dup(2)
os.dup2(_dn, 2)
try:
    import arrow
    from pygame import mixer
finally:
    os.dup2(_se, 2)
    os.close(_se)
    os.close(_dn)


def parse_alarm(inp: str) -> str:
    inp = inp.lower()
    p1 = inp.find("alarm for")
    p_a = inp.find("a.m.")
    p_p = inp.find("p.m.")
    p_c = inp.find(":")

    base = inp[p1 + len("alarm for"):].strip() if p1 != -1 else inp

    if p_a != -1 and p_c != -1:
        t = inp[p1 + len("alarm for") + 1:p_a].strip() + " AM"
    elif p_p != -1 and p_c != -1:
        t = inp[p1 + len("alarm for") + 1:p_p].strip() + " PM"
    elif p_a != -1:
        t = inp[p1 + len("alarm for") + 1:p_a].strip() + ":00 AM"
    elif p_p != -1:
        t = inp[p1 + len("alarm for") + 1:p_p].strip() + ":00 PM"
    else:
        return ""
    return t.strip()


def run_alarm(inp: str):
    alarm_time = parse_alarm(inp)
    if not alarm_time:
        print("SPEAK: Could not understand alarm time.")
        sys.exit(1)

    print(f"SPEAK: Alarm set for {alarm_time}.")
    sys.stdout.flush()

    while True:
        now = arrow.now().format('h:mm A')
        if alarm_time.strip() == now.strip():
            print("SPEAK: Your alarm is going off!")
            sys.stdout.flush()
            try:
                root = Path(__file__).resolve().parent.parent
                alarm_file = root / 'alarm.wav'
                if alarm_file.exists():
                    mixer.init()
                    mixer.music.load(str(alarm_file))
                    mixer.music.play()
                    while mixer.music.get_busy():
                        time.sleep(1)
                else:
                    print("SPEAK: Alarm triggered! No alarm.wav found.")
            except Exception as e:
                print(f"SPEAK: Alarm triggered! Sound error: {e}")
            sys.stdout.flush()
            break
        time.sleep(5)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("SPEAK: No alarm command provided.")
        sys.exit(1)
    run_alarm(' '.join(sys.argv[1:]))
