from __future__ import annotations

import argparse
import json
import shutil
import tarfile
from pathlib import Path


IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare one LCAMAI tar.gz cycle.")
    parser.add_argument("--archive-root", required=True)
    parser.add_argument("--frame-cache-root", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--frame-count", type=int, default=10)
    return parser.parse_args()


def parse_frame_name(path: Path) -> dict[str, str]:
    parts = path.stem.split("-")
    return {
        "date": parts[0],
        "time": parts[1],
        "river_id": parts[2],
        "site_id": parts[3],
        "stem": path.stem,
        "name": path.name,
        "yyyymm": parts[0][:6],
        "video": f"{parts[2]}-{parts[3]}",
    }


def latest_archive(archive_root: Path) -> Path:
    return sorted(archive_root.glob("*.tar.gz"))[-1]


def prune_frames(video_dir: Path, frame_count: int) -> None:
    frames = sorted(p for p in video_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
    for old in frames[:-frame_count]:
        old.unlink()


def main() -> None:
    args = parse_args()
    archive = latest_archive(Path(args.archive_root))
    frame_cache_root = Path(args.frame_cache_root)
    manifest_path = Path(args.manifest)
    staging_dir = manifest_path.parent / "extract"

    if staging_dir.exists():
        shutil.rmtree(staging_dir)
    staging_dir.mkdir(parents=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    frame_cache_root.mkdir(parents=True, exist_ok=True)

    with tarfile.open(archive, "r:gz") as tar:
        tar.extractall(staging_dir)

    items = []
    for extracted in sorted(p for p in staging_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS):
        meta = parse_frame_name(extracted)
        video_dir = frame_cache_root / meta["video"]
        video_dir.mkdir(parents=True, exist_ok=True)
        dst = video_dir / meta["name"]
        shutil.copy2(extracted, dst)
        prune_frames(video_dir, args.frame_count)
        items.append(
            {
                **meta,
                "video_dir": str(video_dir),
                "frame_path": str(dst),
            }
        )

    manifest = {
        "archive": str(archive),
        "archive_name": archive.name,
        "frame_cache_root": str(frame_cache_root),
        "items": items,
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"archive": archive.name, "items": len(items), "manifest": str(manifest_path)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
