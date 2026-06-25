# River Segmentation Pipeline

## 1. Runtime Entry

The deployment entrypoint is:

```bash
bash run_server_seg.sh
```

In Docker, the container starts the same script by default.

The script can run once or continuously. For server deployment, it is expected to run continuously with:

```bash
LOOP=1
INTERVAL_SECONDS=300
```

This means one processing cycle is started every 5 minutes.

## 2. Input Data

The script reads image archives directly from the server filesystem:

```text
/mount/lcamai/latest-all/
```

The input file format is:

```text
YYYYMMDD-HHMM.tar.gz
```

Each tar archive contains river camera images named:

```text
YYYYMMDD-HHMM-河川ID-地点ID.jpg
```

The pipeline does not use `curl`, download, or upload. It only reads local files from `/mount/lcamai/latest-all/` and writes local results to `/mount/lcamai/result-all/`.

## 3. Cycle Preparation

At the beginning of each cycle, `run_server_seg.sh` calls:

```bash
python -m river_seg_server.prepare_lcamai_cycle
```

This module selects the latest `*.tar.gz` archive from `ARCHIVE_ROOT`, extracts it into the runtime directory, parses each image filename, and updates the local rolling frame cache.

The default runtime paths are:

```text
ARCHIVE_ROOT=/mount/lcamai/latest-all
RUNTIME_ROOT=/mount/lcamai/river-seg-runtime
FRAME_CACHE_ROOT=/mount/lcamai/river-seg-runtime/frames
MANIFEST=/mount/lcamai/river-seg-runtime/manifest.json
```

For each image, the video identity is:

```text
河川ID-地点ID
```

For example:

```text
20260625-1200-89094001-111009042.jpg
```

becomes:

```text
video = 89094001-111009042
river_id = 89094001
site_id = 111009042
```

The frame cache keeps only the latest `FRAME_COUNT` frames per video. The default is:

```text
FRAME_COUNT=10
```

Older frames are deleted from the cache.

## 4. Manifest

After extraction and cache update, the preparation step writes:

```text
/mount/lcamai/river-seg-runtime/manifest.json
```

The manifest records the images that need to be processed in the current cycle, including:

```text
date
time
river_id
site_id
video
frame_path
video_dir
```

The segmentation workers consume this manifest instead of scanning arbitrary directories.

## 5. Multi-GPU Worker Split

`run_server_seg.sh` starts one worker process per GPU ID.

Example:

```bash
GPU_IDS="0 1 2 3"
```

starts 4 worker processes.

Each worker loads one SAM3 model copy on its assigned GPU and processes a disjoint subset of the current manifest:

```text
worker 0: items 0, 4, 8, ...
worker 1: items 1, 5, 9, ...
worker 2: items 2, 6, 10, ...
worker 3: items 3, 7, 11, ...
```

This parallelizes across videos. A single video is still processed by one worker.

## 6. Model Loading

Each worker runs:

```bash
python -m river_seg_server.run_segmentation
```

The worker loads:

```text
BASE_CHECKPOINT=/path/to/sam3.pt
CHECKPOINT=/path/to/checkpoint_best.pt
```

`BASE_CHECKPOINT` is the SAM3 base checkpoint.

`CHECKPOINT` is the river segmentation fine-tuned checkpoint.

Both paths are provided at runtime. They are not hard-coded in the code and are not included in the Docker image.

## 7. Per-Video Forward

For each video in the manifest, the worker:

1. Reads the latest `FRAME_COUNT` frames from the rolling frame cache.
2. Builds a temporary SAM3-compatible numeric frame window.
3. Runs SAM3 video propagation.
4. Extracts the mask for the latest frame.
5. Computes mask area in pixels and area ratio.
6. Evaluates alarm status and flood level.
7. Writes the JSON result file.
8. Optionally saves mask and overlay PNG files.

The temporary SAM3 numeric frame window is stored under:

```text
/mount/lcamai/river-seg-runtime/workers/
```

This is runtime data only.

## 8. Threshold And Flood Level

Thresholds are configured by:

```text
THRESHOLD_JSON=config/thresholds.example.json
```

The JSON supports a global default and per-video overrides.

Example:

```json
{
  "default": {
    "area_pixels": 300000,
    "area_ratio": 0.35,
    "flood_levels": [
      { "level": 1, "area_ratio": 0.25 },
      { "level": 2, "area_ratio": 0.35 },
      { "level": 3, "area_ratio": 0.5 }
    ]
  },
  "videos": {
    "89094001-111009042": {
      "area_pixels": 300000,
      "area_ratio": 0.35,
      "flood_levels": [
        { "level": 1, "area_ratio": 0.25 },
        { "level": 2, "area_ratio": 0.35 },
        { "level": 3, "area_ratio": 0.5 }
      ]
    }
  }
}
```

`alarm` becomes `true` if either `area_pixels` or `area_ratio` exceeds its configured threshold.

`flood_level` is assigned from `flood_levels` using the mask area ratio. The highest matched level is used.

## 9. Output Data

The required output root is:

```text
/mount/lcamai/result-all/
```

For an input image:

```text
YYYYMMDD-HHMM-河川ID-地点ID.jpg
```

the result JSON is written to:

```text
/mount/lcamai/result-all/河川ID/地点ID/YYYYMM/YYYYMMDD-HHMM-河川ID-地点ID.txt
```

Example:

```text
/mount/lcamai/result-all/89094001/111009042/202606/20260625-1200-89094001-111009042.txt
```

The `.txt` file contains JSON like:

```json
{
  "frame": "20260625-1200-89094001-111009042.jpg",
  "river_id": "89094001",
  "site_id": "111009042",
  "inference_seconds": 0.42,
  "mask_path": "/mount/lcamai/result-all/89094001/111009042/202606/20260625-1200-89094001-111009042_mask.png",
  "overlay_path": "/mount/lcamai/result-all/89094001/111009042/202606/20260625-1200-89094001-111009042_overlay.png",
  "area_pixels": 350000,
  "area_ratio": 0.41,
  "threshold_checks": [
    {
      "type": "area_pixels",
      "value": 350000,
      "threshold": 300000,
      "alarm": true
    },
    {
      "type": "area_ratio",
      "value": 0.41,
      "threshold": 0.35,
      "alarm": true
    }
  ],
  "alarm": true,
  "flood_level": 2
}
```

If visualization is enabled, mask and overlay files are saved next to the JSON result.

## 10. Runtime Logs

Worker logs and merged runtime summaries are written under:

```text
/mount/lcamai/river-seg-runtime/
```

Important files include:

```text
manifest.json
summary.json
results.jsonl
alarms.jsonl
workers/worker_<index>_gpu_<id>.log
```

These files are for debugging and monitoring. The official result files are the JSON `.txt` files under `/mount/lcamai/result-all/`.

## 11. Docker Deployment Command

Typical long-running Docker command:

```bash
docker run -d --name river-seg \
  --restart unless-stopped \
  --gpus all --shm-size 16g \
  -v /mount/lcamai:/mount/lcamai \
  -v /path/to/checkpoints:/checkpoints:ro \
  -e BASE_CHECKPOINT=/checkpoints/sam3.pt \
  -e CHECKPOINT=/checkpoints/checkpoint_best.pt \
  -e ARCHIVE_ROOT=/mount/lcamai/latest-all \
  -e OUTPUT_ROOT=/mount/lcamai/result-all \
  -e RUNTIME_ROOT=/mount/lcamai/river-seg-runtime \
  -e THRESHOLD_JSON=/workspace/river_seg_server_deploy/config/thresholds.example.json \
  -e GPU_IDS="0 1 2 3" \
  -e FRAME_COUNT=10 \
  -e SAVE_MASK=1 \
  -e SAVE_OVERLAY=1 \
  -e LOOP=1 \
  -e INTERVAL_SECONDS=300 \
  river-seg-sam3:latest
```
