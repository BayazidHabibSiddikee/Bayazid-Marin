#!/usr/bin/env python3
# tools/timer.py — runs as its own process
# Usage: python timer.py "timer for 1 hour 30 minutes"
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


def parse_command(inp: str):
    inp = inp.lower()
    p1 = inp.find('timer for')
    p_h = inp.find('hour')
    p_m = inp.find('minute')

    if p_h == -1 and p_m == -1:
        print("SPEAK: Could not understand timer command.")
        sys.exit(1)

    addhour = 0
    addmin = 0

    if p_h != -1 and p_m == -1:
        addhour = int(inp[p1 + len('timer for'):p_h].strip() or 0)
    elif p_h == -1 and p_m != -1:
        addmin = int(inp[p1 + len('timer for'):p_m].strip() or 0)
    else:
        addhour = int(inp[p1 + len('timer for'):p_h].strip() or 0)
        addmin  = int(inp[p_h + len('hour'):p_m].strip() or 0)

    return addhour, addmin


def run_timer(inp: str):
    addhour, addmin = parse_command(inp)
    now = arrow.now()
    target = now.shift(hours=addhour, minutes=addmin)
    end_str = target.format('H:m:s')

    print(f"SPEAK: Timer set for {addhour} hour(s) and {addmin} minute(s). "
          f"Goes off at {target.format('h:mm A')}.")
    sys.stdout.flush()

    while True:
        if arrow.now().format('H:m:s') == end_str:
            print("SPEAK: Time's up!")
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
                    print("SPEAK: Timer done! No alarm.wav found.")
            except Exception as e:
                print(f"SPEAK: Timer done! Sound error: {e}")
                sys.stdout.flush()
            break
        time.sleep(1)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("SPEAK: No timer command provided.")
        sys.exit(1)
    run_timer(' '.join(sys.argv[1:]))
