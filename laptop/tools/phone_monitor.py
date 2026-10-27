"""M2: watch the phone link live — rate, stalls, jitter, pointer angles, compass, GPS, buttons.

    python tools/phone_monitor.py [--port 8765] [--seconds 60]

Prints one line per second and a summary at the end (use it to fill in the wired vs Wi-Fi table in
docs/hardware-checklist.md). Pointer yaw is in the phone's own frame (arbitrary zero) until Align.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pointview import world
from pointview.phone_link import PhoneLinkThread


def fmt(x, spec: str = ".1f", none: str = "—") -> str:
    return none if x is None else format(x, spec)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--seconds", type=float, default=0, help="0 = until Ctrl+C")
    a = p.parse_args(argv)

    link = PhoneLinkThread(host=a.host, port=a.port)
    link.start()
    print(f"listening on {a.host}:{link.port} — open PointView on the phone", flush=True)
    t0 = time.monotonic()
    worst_excess = 0.0
    try:
        while not a.seconds or time.monotonic() - t0 < a.seconds:
            time.sleep(1.0)
            s = link.stats()
            st = link.latest()
            buttons = [b.name for b in link.pop_buttons()]
            if s.excess_latency_ms is not None:
                worst_excess = max(worst_excess, s.excess_latency_ms)
            if st is None:
                print(f"connected={s.connected} waiting for samples", flush=True)
                continue
            yaw, pitch = world.yaw_pitch_from_vector(world.phone_pointer_vector(st.quat))
            print(
                f"{s.rate_hz:5.1f} Hz  stalls {s.stalls:3d}  max gap {s.longest_gap_ms:6.1f} ms  "
                f"jitter {fmt(s.excess_latency_ms)} ms | ptr yaw {yaw:6.1f} pitch {pitch:+5.1f}  "
                f"ω {tuple(round(w, 2) for w in st.omega)} | heading {fmt(st.heading_deg)}±"
                f"{fmt(st.heading_acc, '.0f')}  GPS ±{fmt(st.gps_acc, '.0f')} m  {' '.join(buttons)}",
                flush=True,
            )
    except KeyboardInterrupt:
        pass
    finally:
        s = link.stats()
        link.stop()
        print(
            f"\nsummary: messages {s.messages}, invalid {s.invalid}, stalls (>50 ms) {s.stalls}, "
            f"longest gap {s.longest_gap_ms:.1f} ms, worst jitter {worst_excess:.1f} ms, "
            f"connections {s.connections}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
