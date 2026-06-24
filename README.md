# River Seg Server Deploy

## 1. Configure Environment

Conda:

```bash
conda env create -f river_seg_server_deploy/environment.yml
```

Docker:

```bash
cd river_seg_server_deploy
docker build -t river-seg-sam3:latest .
```


## 2. Run With Conda

One-shot run:

```bash
cd <repo-root>

INPUT_ROOT=/path/to/local/frames \
OUTPUT_ROOT=/path/to/output \
THRESHOLD_JSON=/path/to/thresholds.json \
BASE_CHECKPOINT=<repo-root>/river_seg_server_deploy/river_seg_server/checkpoint/sam3.pt \
CHECKPOINT=<repo-root>/river_seg_server_deploy/river_seg_server/checkpoint/checkpoint_best.pt \
GPU_COUNT=1 \
FRAME_COUNT=10 \
SAVE_MASK=1 \
SAVE_OVERLAY=1 \
bash river_seg_server_deploy/run_server_seg.sh
```

Multi-GPU run:

```bash
cd <repo-root>

GPU_IDS="0 1 2 3" \
INPUT_ROOT=/path/to/local/frames \
OUTPUT_ROOT=/path/to/output \
THRESHOLD_JSON=/path/to/thresholds.json \
BASE_CHECKPOINT=<repo-root>/river_seg_server_deploy/river_seg_server/checkpoint/sam3.pt \
CHECKPOINT=<repo-root>/river_seg_server_deploy/river_seg_server/checkpoint/checkpoint_best.pt \
FRAME_COUNT=10 \
SAVE_MASK=1 \
SAVE_OVERLAY=1 \
bash river_seg_server_deploy/run_server_seg.sh
```

This starts one worker process per GPU ID. Each worker loads one model copy on
its assigned GPU and processes a disjoint subset of videos.

Run every 5 minutes:

```bash
cd <repo-root>

LOOP=1 \
INTERVAL_SECONDS=300 \
GPU_IDS="0 1 2 3" \
INPUT_ROOT=/path/to/local/frames \
OUTPUT_ROOT=/path/to/output \
THRESHOLD_JSON=/path/to/thresholds.json \
BASE_CHECKPOINT=<repo-root>/river_seg_server_deploy/river_seg_server/checkpoint/sam3.pt \
CHECKPOINT=<repo-root>/river_seg_server_deploy/river_seg_server/checkpoint/checkpoint_best.pt \
FRAME_COUNT=10 \
SAVE_MASK=1 \
SAVE_OVERLAY=1 \
bash river_seg_server_deploy/run_server_seg.sh
```

## 3. Parameters

| Parameter | Default | Description |
| --- | --- | --- |
| `PYTHON_BIN` | `python` | Python executable |
| `PROJECT_ROOT` | parent directory of `river_seg_server_deploy` | Root path of the main project |
| `BASE_CHECKPOINT` | required | SAM3 base model |
| `CHECKPOINT` | required | River segmentation checkpoint |
| `INPUT_ROOT` | `PROJECT_ROOT/frames/original` | Local input frame directory |
| `OUTPUT_ROOT` | `PROJECT_ROOT/outputs/server_deploy_run` | Output directory |
| `THRESHOLD_JSON` | `config/thresholds.example.json` | Threshold config file |
| `GPU_COUNT` | `1` | Number of workers if `GPU_IDS` is not set, using IDs `0..GPU_COUNT-1` |
| `GPU_IDS` | empty | Explicit GPU IDs and worker list, e.g. `"0 1 2 3"` |
| `FRAME_COUNT` | `10` | Number of latest frames per video |
| `SAVE_MASK` | `1` | `1` saves mask PNG, `0` disables it |
| `SAVE_OVERLAY` | `1` | `1` saves overlay PNG, `0` disables it |
| `LIMIT` | `0` | Limit number of videos; `0` means no limit |
| `LOOP` | `0` | `1` runs continuously |
| `INTERVAL_SECONDS` | `300` | Loop interval in seconds |
| `MAX_CYCLES` | `0` | Stop after N cycles; `0` means unlimited |

For fail-fast behavior, if any worker exits with an error, the cycle exits
non-zero. Worker logs are written to:

```text
OUTPUT_ROOT/workers/worker_<index>_gpu_<id>.log
```

Threshold JSON example:

```json
{
  "default": {
    "area_pixels": 300000,
    "area_ratio": 0.35
  },
  "videos": {
    "89094001-111009042": {
      "area_pixels": 300000,
      "area_ratio": 0.35
    }
  }
}
```

If either `area_pixels` or `area_ratio` is exceeded, the video is marked as an alarm.


## 4. Run With Docker

One-shot run:

```bash
cd river_seg_server_deploy

docker run --rm --gpus all --shm-size 16g \
  -v /path/to/frames:/data/frames:ro \
  -v /path/to/outputs:/data/outputs \
  -v /path/to/checkpoints:/checkpoints:ro \
  -e INPUT_ROOT=/data/frames \
  -e OUTPUT_ROOT=/data/outputs \
  -e THRESHOLD_JSON=/workspace/river_seg_server_deploy/config/thresholds.example.json \
  -e BASE_CHECKPOINT=/checkpoints/sam3.pt \
  -e CHECKPOINT=/checkpoints/checkpoint_best.pt \
  -e GPU_IDS="0" \
  -e FRAME_COUNT=10 \
  -e SAVE_MASK=1 \
  -e SAVE_OVERLAY=1 \
  river-seg-sam3:latest
```

Run every 5 minutes:

```bash
cd river_seg_server_deploy

docker run --rm --gpus all --shm-size 16g \
  -v /path/to/frames:/data/frames:ro \
  -v /path/to/outputs:/data/outputs \
  -v /path/to/checkpoints:/checkpoints:ro \
  -e INPUT_ROOT=/data/frames \
  -e OUTPUT_ROOT=/data/outputs \
  -e THRESHOLD_JSON=/workspace/river_seg_server_deploy/config/thresholds.example.json \
  -e BASE_CHECKPOINT=/checkpoints/sam3.pt \
  -e CHECKPOINT=/checkpoints/checkpoint_best.pt \
  -e GPU_IDS="0 1 2 3" \
  -e FRAME_COUNT=10 \
  -e SAVE_MASK=1 \
  -e SAVE_OVERLAY=1 \
  -e LOOP=1 \
  -e INTERVAL_SECONDS=300 \
  river-seg-sam3:latest
```

The checkpoint files are mounted at runtime and are not included in the Docker
image.
