"""Run PointView: ``python -m pointview`` (real glasses) or ``python -m pointview --mock``.

The Qt event loop owns the main thread; a 60 Hz timer calls PointViewApp.tick(). The phone link runs on
its own asyncio thread, the detector and landmark lookups on their own threads.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import sys
import tempfile
import time
from dataclasses import asdict
from pathlib import Path

from pointview.app import PointViewApp, RunStats
from pointview.interfaces import InfoCard, Layer

log = logging.getLogger("pointview")

DEFAULT_MODEL = Path(__file__).resolve().parents[1] / "models" / "rtdetr_v2_r18vd.onnx"


def default_cache() -> Path:
    base = os.environ.get("LOCALAPPDATA") or os.path.expanduser("~/.cache")
    return Path(base) / "PointView" / "cache.sqlite"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="pointview", description=__doc__.splitlines()[0])
    p.add_argument(
        "--mock", action="store_true", help="mock glasses + fake phone + fixture landmarks"
    )
    p.add_argument("--video", type=Path, help="mock: play this video instead of the fixture")
    p.add_argument(
        "--live-landmarks", action="store_true", help="mock: query real Overpass/Wikipedia"
    )
    p.add_argument("--duration", type=float, help="quit after this many seconds")
    p.add_argument("--fps", type=float, default=60.0)
    p.add_argument("--port", type=int, default=8765, help="phone link port (mock uses a free port)")
    p.add_argument("--screen", help="glasses display name substring, e.g. VITURE")
    p.add_argument("--screen-index", type=int)
    p.add_argument(
        "--camera-index", type=int, help="OpenCV camera index instead of SDK pass-through"
    )
    p.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    p.add_argument("--no-detection", action="store_true")
    p.add_argument("--no-landmarks", action="store_true")
    p.add_argument("--cache", type=Path, help="landmark SQLite cache")
    p.add_argument("--screenshot-dir", type=Path, help="save PNGs of what is drawn")
    p.add_argument(
        "--screenshot-every", type=float, default=2.0, help="seconds between screenshots"
    )
    p.add_argument("--stats-json", type=Path, help="write the run counters here on exit")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args(argv)


def run(args: argparse.Namespace) -> RunStats:
    """Build everything, run the Qt loop until --duration or Ctrl+C, and return the run counters."""
    from PySide6.QtCore import QTimer

    from pointview import render
    from pointview.landmarks import LandmarkLayer
    from pointview.phone_link import PhoneLinkThread

    qt = render.ensure_app()
    layers: list[Layer] = []

    if args.mock:
        from pointview import mock

        glasses = mock.make_glasses(args.video)
        phone = PhoneLinkThread(host="127.0.0.1", port=0)
    else:
        from pointview.glasses import VitureGlasses

        glasses = VitureGlasses(camera_index=args.camera_index)
        phone = PhoneLinkThread(port=args.port)

    if not args.no_detection:
        if args.model.exists():
            from pointview.detection import DetectionLayer

            layers.append(DetectionLayer(args.model))
        else:
            log.warning("no model at %s: run tools/download_model.py. Detection off.", args.model)

    landmark_layer = None
    if not args.no_landmarks:
        client = None
        cache = args.cache or default_cache()
        if args.mock and not args.live_landmarks:
            from pointview import mock

            client = mock.landmark_client()
            cache = args.cache or Path(tempfile.mkdtemp(prefix="pointview-mock-")) / "cache.sqlite"
        landmark_layer = LandmarkLayer(cache, client=client)
        layers.append(landmark_layer)

    renderer = render.QtRenderer(args.screen, args.screen_index, glasses.display_size, pump=False)
    app = PointViewApp(glasses, layers, phone, renderer, landmark_layer=landmark_layer)

    glasses.start()
    phone.start()
    log.info("phone link listening on port %d", phone.port)
    fake = None
    if args.mock:
        from pointview import mock

        fake = mock.FakePhoneThread(f"ws://127.0.0.1:{phone.port}")
        fake.start()

    shots = args.screenshot_dir
    if shots:
        shots.mkdir(parents=True, exist_ok=True)
    t_start = time.monotonic()
    next_shot = [0.0]
    saved_info = [0]

    def tick() -> None:
        items = app.tick()
        if shots is None:
            return
        elapsed = time.monotonic() - t_start
        if elapsed >= next_shot[0]:
            render.save_png(items, shots / f"frame_{elapsed:05.1f}s.png", glasses.display_size)
            next_shot[0] += args.screenshot_every
        # The first card, then again once it shows its Wikipedia summary (fetched in the background).
        cards = [i for i in items if isinstance(i, InfoCard)]
        if cards and (not saved_info[0] or (saved_info[0] == 1 and cards[0].footer)):
            render.save_png(items, shots / "info_card.png", glasses.display_size)
            saved_info[0] = 2 if cards[0].footer else 1

    timer = QTimer()
    timer.setInterval(max(1, round(1000 / args.fps)))
    timer.timeout.connect(tick)
    timer.start()
    if args.duration:
        QTimer.singleShot(round(args.duration * 1000), qt.quit)
    # Let Ctrl+C through: Python only handles signals when it gets control back from Qt.
    signal.signal(signal.SIGINT, lambda *_: qt.quit())
    wake = QTimer()
    wake.timeout.connect(lambda: None)
    wake.start(200)

    try:
        qt.exec()
    finally:
        timer.stop()
        wake.stop()
        app.close()
        phone.stop()
        glasses.stop()
        renderer.close()
        if fake is not None and fake.error is not None:
            log.error("fake phone failed: %r", fake.error)
    if shots:
        render.save_png(app.items, shots / "last.png", glasses.display_size)
    return app.stats


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    if args.mock and not os.environ.get("DISPLAY") and sys.platform.startswith("linux"):
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    stats = run(args)
    log.info(
        "frames=%d fps=%.1f max_entities=%d kinds=%s snaps=%d aligns=%d info_frames=%d",
        stats.frames,
        stats.fps,
        stats.max_entities,
        sorted(stats.kinds_seen),
        stats.snaps,
        stats.aligns,
        stats.info_frames,
    )
    if args.stats_json:
        data = asdict(stats)
        data["kinds_seen"] = sorted(stats.kinds_seen)
        data["info_titles"] = sorted(stats.info_titles)
        args.stats_json.write_text(json.dumps(data, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
