# River Seg Server Deploy

## 1. Build Environment

Conda:

```bash
conda env create -f environment.yml
```

Docker:

```bash
docker build -t river-seg-sam3:latest .
```

## 2. Download Checkpoints

Expected checkpoint paths:

```text
river_seg_server/checkpoint/sam3.pt
river_seg_server/checkpoint/checkpoint_best.pt
```

Download with `gdown`:

```bash
python -m pip install gdown
mkdir -p river_seg_server/checkpoint

gdown "https://drive.google.com/uc?export=download&id=1hZOI-mZPxFnhmie7IkjssZf3DfGkKyQy" -O river_seg_server/checkpoint/sam3.pt
gdown "https://drive.google.com/uc?export=download&id=1LwqPLxTvoWKhuEWoM5ApSPzmDDmLRmkU" -O river_seg_server/checkpoint/checkpoint_best.pt
```

Download with `curl`:

```bash
mkdir -p river_seg_server/checkpoint

curl -L "https://drive.google.com/uc?export=download&id=1hZOI-mZPxFnhmie7IkjssZf3DfGkKyQy" -o river_seg_server/checkpoint/sam3.pt
curl -L "https://drive.google.com/uc?export=download&id=1LwqPLxTvoWKhuEWoM5ApSPzmDDmLRmkU" -o river_seg_server/checkpoint/checkpoint_best.pt
```

The links are mapped in the order provided. If the files were provided in the opposite order, swap the two output names.

## 3. Data Layout

Input archives are read from:

```text
/mount/lcamai/latest-all/YYYYMMDD-HHMM.tar.gz
```

Each archive contains images named:

```text
YYYYMMDD-HHMM-河川ID-地点ID.jpg
```

Results are written to:

```text
/mount/lcamai/result-all/河川ID/地点ID/YYYYMM/YYYYMMDD-HHMM-河川ID-地点ID.txt
```

The `.txt` file is JSON and includes `flood_level`.

## 4. Run With Conda

```bash
BASE_CHECKPOINT=/path/to/sam3.pt \
CHECKPOINT=/path/to/checkpoint_best.pt \
ARCHIVE_ROOT=/mount/lcamai/latest-all \
OUTPUT_ROOT=/mount/lcamai/result-all \
RUNTIME_ROOT=/mount/lcamai/river-seg-runtime \
THRESHOLD_JSON=config/thresholds.example.json \
GPU_IDS="0 1 2 3" \
FRAME_COUNT=10 \
SAVE_MASK=1 \
SAVE_OVERLAY=1 \
LOOP=1 \
INTERVAL_SECONDS=300 \
bash run_server_seg.sh
```

## 5. Run With Docker

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

## 6. Parameters

| Parameter | Default | Description |
| --- | --- | --- |
| `BASE_CHECKPOINT` | required | SAM3 base checkpoint |
| `CHECKPOINT` | required | River segmentation checkpoint |
| `ARCHIVE_ROOT` | `/mount/lcamai/latest-all` | Input tar.gz directory |
| `OUTPUT_ROOT` | `/mount/lcamai/result-all` | Required JSON result directory |
| `RUNTIME_ROOT` | `/mount/lcamai/river-seg-runtime` | Rolling frame cache, manifests, worker logs |
| `THRESHOLD_JSON` | `config/thresholds.example.json` | Area and flood-level thresholds |
| `GPU_IDS` | empty | Explicit worker GPU IDs, e.g. `"0 1 2 3"` |
| `GPU_COUNT` | `1` | Used only when `GPU_IDS` is empty |
| `FRAME_COUNT` | `10` | Number of latest frames kept per video |
| `SAVE_MASK` | `1` | Save mask PNG next to the result JSON |
| `SAVE_OVERLAY` | `1` | Save overlay PNG next to the result JSON |
| `LIMIT` | `0` | Limit processed videos; `0` means no limit |
| `LOOP` | `0` | `1` runs continuously |
| `INTERVAL_SECONDS` | `300` | Loop interval |
| `MAX_CYCLES` | `0` | Stop after N cycles; `0` means unlimited |

Threshold JSON:

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
