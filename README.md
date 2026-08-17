# Dynamic Braking System

Perception-driven braking for a simulated autonomous vehicle in [CARLA](https://carla.org/).

The vehicle detects objects from a front-facing camera, estimates their range, and
uses that range to modulate throttle and brake in a closed loop. Detection started on
MobileNet-SSD and was later moved to YOLOv10; both stages are kept here so the
progression — and the calibration work each one needed — stays visible.

## What this repository covers

- **Closed-loop control in CARLA** — ego vehicle driven in synchronous mode, with
  throttle and brake derived from live detections rather than a fixed script.
- **Two detection backends** — MobileNet-SSD (`ssd_mobilenet_v3_large_coco`) first,
  then YOLOv10, so the two can be compared on the same scenes.
- **Monocular range estimation** — pinhole distance estimation from bounding-box
  geometry, plus the camera calibration and sensor-separation work it depends on.
- **Traffic light state** — a custom-labelled traffic light dataset (`custom_dataset/`)
  trained into `best.pt`, so the controller reacts to light colour, not just presence.
- **Braking profile comparison** — `Carla/s_curve_chart.py` plots constant-deceleration
  against an S-curve profile over the same stopping distance.

## Layout

| Path | Contents |
|---|---|
| `Carla/` | CARLA integration: object detection, distance estimation, braking profile chart |
| `Object_Detection_with_MobileNet/` | Standalone MobileNet-SSD detection on image, video, and camera input |
| `custom_dataset/` | Traffic light images and the labelled YOLOv8-format export used for training |
| `object_tracking_*.py` | YOLOv10 tracking over camera input, a video file, and COCO classes |
| `source/`, `output_image/` | Test inputs and saved detection output |
| `best.pt` | Traffic light model trained from `custom_dataset/` |
| `Note.txt` | Open design questions carried forward |

## Requirements

- CARLA 0.9.x with its Python API on `PYTHONPATH`
- Python 3.8+
- `opencv-python`, `numpy`, `matplotlib`
- `ultralytics` (provides `YOLOv10`) — see [ultralytics/yolov10](https://github.com/THU-MIG/yolov10)

Pretrained YOLOv10 checkpoints (`yolov10n/s/m/b.pt`) are **not** committed — `ultralytics`
downloads them on first use. `best.pt` is committed because it is trained from this
repository's own dataset.

## Running

Start a CARLA server, then:

```bash
# MobileNet-SSD detection inside CARLA
python Carla/Object_detection_with_Mobilenet.py

# MobileNet-SSD detection plus distance estimation
python Carla/Distance_estimate_mobilenetssd_carla.py

# YOLOv10 traffic light tracking over a recorded video
python object_tracking_within_video.py

# Braking profile comparison
python Carla/s_curve_chart.py
```

The MobileNet scripts read `frozen_inference_graph.pb` and the matching `.pbtxt` from
their own directory, so run them from the repository root as shown.

## Status

Thesis work, still active. `Note.txt` tracks the next open item: a safety buffer that
tightens to a minimum for pedestrians, so no cyclist can occupy the gap ahead of the
vehicle.

## License

MIT — see [LICENSE](LICENSE).
