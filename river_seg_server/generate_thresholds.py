from __future__ import annotations

import argparse
import json
from pathlib import Path


IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


def has_images(path: Path) -> bool:
    return any(p.is_file() and p.suffix.lower() in IMAGE_EXTS for p in path.iterdir())


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate threshold JSON from video directory names.")
    p.add_argument("--frames-root", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--area-pixels", type=float, default=300000)
    p.add_argument("--area-ratio", type=float, default=0.35)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    frames_root = Path(args.frames_root)
    videos = {}
    for path in sorted(frames_root.iterdir()):
        if path.is_dir() and has_images(path):
            videos[path.name] = {
                "area_pixels": args.area_pixels,
                "area_ratio": args.area_ratio,
            }
    payload = {
        "default": {
            "area_pixels": args.area_pixels,
            "area_ratio": args.area_ratio,
        },
        "videos": videos,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "videos": len(videos)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
