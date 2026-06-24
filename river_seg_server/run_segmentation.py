from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch


IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one server-local river segmentation cycle.")
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--threshold-json", required=True)
    parser.add_argument("--base-checkpoint", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--text-prompt", default="water")
    parser.add_argument("--gpus", type=int, nargs="+", default=[0])
    parser.add_argument("--frame-count", type=int, default=10)
    parser.add_argument("--prompt-frame", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--worker-index", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--save-mask", action="store_true")
    parser.add_argument("--save-overlay", action="store_true")
    return parser.parse_args()


def add_project_paths() -> None:
    deploy_root = Path(__file__).resolve().parents[1]
    project_root = Path(os.environ["PROJECT_ROOT"]).resolve()
    for path in (deploy_root, project_root, project_root / "sam3"):
        sys.path.insert(0, str(path))


add_project_paths()

from sam3.model_builder import build_sam3_video_predictor  # noqa: E402
from sam3_river_finetune.eval_original_video_tracking import (  # noqa: E402
    add_initial_prompt,
    extract_primary_mask_from_sam3_outputs,
    install_finetuned_detector,
    load_checkpoint_config,
    load_sorted_frame_paths,
    read_frame_shape,
    save_mask_and_overlay,
)


def image_frames(video_dir: Path) -> list[Path]:
    frames = [p for p in video_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
    return sorted(frames, key=lambda p: (p.stem, p.name))


def video_dirs(input_root: Path, limit: int, worker_index: int, num_workers: int) -> list[Path]:
    videos = [p for p in sorted(input_root.iterdir()) if p.is_dir() and image_frames(p)]
    if limit > 0:
        videos = videos[:limit]
    return videos[worker_index::num_workers]


def build_numeric_window(video_dir: Path, windows_root: Path, frame_count: int) -> tuple[Path, list[dict[str, Any]]]:
    frames = image_frames(video_dir)[-frame_count:]
    window_dir = windows_root / video_dir.name
    window_dir.mkdir(parents=True, exist_ok=True)
    for old in window_dir.iterdir():
        old.unlink()

    mapping = []
    for idx, src in enumerate(frames):
        name = f"{idx:06d}.jpg"
        dst = window_dir / name
        dst.symlink_to(src.resolve())
        mapping.append({"frame_index": idx, "sam3_name": name, "source_name": src.name, "source_path": str(src)})

    (window_dir / "frame_mapping.json").write_text(json.dumps(mapping, ensure_ascii=False, indent=2), encoding="utf-8")
    return window_dir, mapping


def load_thresholds(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def video_threshold(thresholds: dict[str, Any], video_name: str) -> dict[str, float]:
    item = thresholds.get("videos", {}).get(video_name, thresholds.get("default", {}))
    return {
        "area_pixels": float(item["area_pixels"]) if "area_pixels" in item else None,
        "area_ratio": float(item["area_ratio"]) if "area_ratio" in item else None,
    }


def load_model(args: argparse.Namespace) -> tuple[Any, dict[str, Any]]:
    config = SimpleNamespace(
        native=False,
        checkpoint=args.checkpoint,
        base_checkpoint=args.base_checkpoint,
        bpe_path=None,
        text_prompt=args.text_prompt,
        trainable=None,
        lora_rank=None,
        lora_alpha=None,
        lora_dropout=None,
        lora_targets=None,
        strict_detector_load=False,
    )
    config, checkpoint = load_checkpoint_config(config)
    predictor = build_sam3_video_predictor(
        checkpoint_path=config.base_checkpoint,
        bpe_path=config.bpe_path,
        strict_state_dict_loading=True,
        gpus_to_use=args.gpus,
    )
    load_summary = install_finetuned_detector(predictor, config, checkpoint)
    return predictor, load_summary


def cuda_sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def forward_video(predictor: Any, window_dir: Path, prompt_frame: int, text_prompt: str) -> tuple[Path, np.ndarray, float]:
    frame_paths = load_sorted_frame_paths(window_dir)
    latest_index = len(frame_paths) - 1
    prompt_index = max(0, min(prompt_frame, latest_index))

    response = predictor.handle_request(request={"type": "start_session", "resource_path": str(window_dir)})
    session_id = response["session_id"]
    try:
        add_initial_prompt(predictor, session_id, prompt_index, text_prompt, None, frame_paths[prompt_index])
        outputs: dict[int, Any] = {}
        request = {"type": "propagate_in_video", "session_id": session_id, "propagation_direction": "both"}
        cuda_sync()
        start = time.perf_counter()
        for item in predictor.handle_stream_request(request=request):
            outputs[int(item["frame_index"])] = item["outputs"]
        cuda_sync()
        inference_seconds = time.perf_counter() - start
    finally:
        predictor.handle_request(request={"type": "close_session", "session_id": session_id})

    frame_path = frame_paths[latest_index]
    height, width = read_frame_shape(frame_path)
    mask = extract_primary_mask_from_sam3_outputs(outputs.get(latest_index, {}), height=height, width=width)
    return frame_path, mask, inference_seconds


def evaluate_mask(mask: np.ndarray, threshold: dict[str, float]) -> dict[str, Any]:
    area_pixels = int((mask > 0).sum())
    area_ratio = area_pixels / float(mask.shape[0] * mask.shape[1])
    checks = []
    if threshold["area_pixels"] is not None:
        checks.append(
            {
                "type": "area_pixels",
                "value": area_pixels,
                "threshold": threshold["area_pixels"],
                "alarm": area_pixels > threshold["area_pixels"],
            }
        )
    if threshold["area_ratio"] is not None:
        checks.append(
            {
                "type": "area_ratio",
                "value": area_ratio,
                "threshold": threshold["area_ratio"],
                "alarm": area_ratio > threshold["area_ratio"],
            }
        )
    return {"area_pixels": area_pixels, "area_ratio": area_ratio, "threshold_checks": checks, "alarm": any(x["alarm"] for x in checks)}


def save_result(
    output_root: Path,
    video_name: str,
    source_frame_name: str,
    mask: np.ndarray,
    frame_path: Path,
    metrics: dict[str, Any],
    inference_seconds: float,
    save_mask: bool,
    save_overlay: bool,
) -> dict[str, Any]:
    result_dir = output_root / "results" / video_name
    result_dir.mkdir(parents=True, exist_ok=True)
    mask_path = result_dir / "mask.png" if save_mask else None
    overlay_path = result_dir / "overlay.png" if save_overlay else None
    if save_mask or save_overlay:
        save_mask_and_overlay(mask, frame_path, mask_path, overlay_path)

    record = {
        "video": video_name,
        "frame": source_frame_name,
        "inference_seconds": inference_seconds,
        "mask_path": str(mask_path) if mask_path else "",
        "overlay_path": str(overlay_path) if overlay_path else "",
        **metrics,
    }
    (result_dir / "summary.json").write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    return record


def append_jsonl(path: Path, item: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    input_root = Path(args.input_root)
    output_root = Path(args.output_root)
    windows_root = output_root / "windows"
    thresholds = load_thresholds(Path(args.threshold_json))

    output_root.mkdir(parents=True, exist_ok=True)
    windows_root.mkdir(parents=True, exist_ok=True)
    for path in (output_root / "results.jsonl", output_root / "alarms.jsonl"):
        if path.exists():
            path.unlink()
        path.touch()

    predictor, load_summary = load_model(args)
    records = []
    start = time.perf_counter()
    try:
        for video_dir in video_dirs(input_root, args.limit, args.worker_index, args.num_workers):
            window_dir, mapping = build_numeric_window(video_dir, windows_root, args.frame_count)
            frame_path, mask, inference_seconds = forward_video(predictor, window_dir, args.prompt_frame, args.text_prompt)
            latest_source = mapping[-1]["source_name"]
            metrics = evaluate_mask(mask, video_threshold(thresholds, video_dir.name))
            record = save_result(
                output_root,
                video_dir.name,
                latest_source,
                mask,
                frame_path,
                metrics,
                inference_seconds,
                args.save_mask,
                args.save_overlay,
            )
            append_jsonl(output_root / "results.jsonl", record)
            if record["alarm"]:
                append_jsonl(output_root / "alarms.jsonl", record)
            print(json.dumps(record, ensure_ascii=False), flush=True)
            records.append(record)
    finally:
        if hasattr(predictor, "shutdown"):
            predictor.shutdown()

    summary = {
        "input_root": str(input_root),
        "output_root": str(output_root),
        "load": load_summary,
        "worker_index": args.worker_index,
        "num_workers": args.num_workers,
        "videos": len(records),
        "alarms": sum(1 for x in records if x["alarm"]),
        "total_seconds": time.perf_counter() - start,
    }
    (output_root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
