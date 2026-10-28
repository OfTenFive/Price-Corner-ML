# Price Corner - TCG Scanner, On-Device TCG Recognition Models & Training Pipeline
Price Corner identifies cards on-device using local ML models, then links straight to Cardmarket, TCGPlayer, and PriceCharting (plus TCGCollector, since I use it for my own collection) instead of showing a single proprietary price estimate. I built it because I wanted a scanner that pointed to prices vendors at conventions/events actually trusted, and because I wanted to build an on-device ML pipeline end to end.

## Features

- **Language support** - English is the officially supported and tested language. The OCR engine has built-in recognition for Japanese, Korean, and Chinese (Simplified & Traditional) as well and other. The card database currently includes ENglish, Japanese and Chinese (Traditional) entries - with room to expand further down the line.
- **Instant on-device scanning** - card detection and OCR run locally, no network round-trip needed to identify a card
- **Real marketplace links, not estimates** - jumps straight to live Cardmarket / TCGPlayer / PriceCharting / listings for the identified card
- **No account, no sign-up, no scan limits**
- **Quick re-check, not a collection tracker** - star a card and it's remembered after you close the app, so you can look up its price again later without rescanning (search box included to find it by name). Not meant for organizing or managing large collections.
- **Card Lookup** - there is a fallback to card lookup option if for whatever reason the scanner refuses to work.

Here is short [demo](https://youtu.be/6YLQzMhJt94) if you want to see it in action.

> *Looking for the Android App?* **Play Store**: [link once published] 

It's free and maintained in my spare time. If you'd like to support it, there's a [Ko-fi link](https://ko-fi.com/oftenfive) - never required, always appreciated.

## Repository Content

This repository contains the full machine learning and synthetic dataset generation pipeline powering **Price Corner - TCG Scanner**, a fast, on-device Pokémon TCG scanner for Android.

It provides everything needed to generate synthetic training data, train custom **YOLO Pose** (card corner detection) and **YOLO OBB** (for card name/number) models, and export them to **LiteRT / TFLite** for real-time mobile inference. The core app uses these pre-trained models on-device alongside PP-OCRv6 tiny and OpenCV to match cards instantly without external network calls.


## Installation and Usage
### Installation
```bash
git clone [https://github.com/OfTenFive/price-corner-ml.git](https://github.com/OfTenFive/price-corner-ml.git)
cd price-corner-ai
pip install -r requirements.txt
```

### Usage
1. Place the card images inside **Card_Scans/** with TCG card scans.
2. Place background scenes inside **backgrounds/**
3. Change the **CANVAS_SIZE** and **SPLITS** values in dataset_generator.py
4. Run the generator:
```bash
# Usage: python dataset_generator.py [num_workers] [chunksize]
python .\dataset_generator.py 8 25
```

### What the Generator Does

- **Perspective Warping:** Randomly scales, rotates ($\pm 35^\circ$), and applies 3D perspective distortion.
- **Lighting & Reflections:** Simulates glare streaks across sleeves/cards, ambient brightness shifts, and out-of-focus camera blurs.
- **Camera Noise:** Adds Gaussian sensor noise and dynamic drop shadows.
- **Auto-Labeling:** Computes exact keypoints for top-left, top-right, bottom-right, and bottom-left corners in YOLO Pose format.

## Model Training
Training is handled via standard [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) scripts.

### Train Card Pose Model (Corner Detection)
```python
from ultralytics import YOLO
model = YOLO("yolo26n-pose.pt") # or custom pose baseline
model.train(
    data="dataset_pose_640/data.yaml",
    epochs=5, # quick test run, change to train real model
    imgsz=640,
    batch=16, # fit valuse to your hardware
    workers=8, # fit valuse to your hardware
    device=0,
    patience=25,
    optimizer="auto",
    project="runs",
    name="card_pose_model"
)
```
### Train Name/Number OBB Model
```python
from ultralytics import YOLO
model = YOLO("yolo26n-obb.pt") # or custom obb baseline
model.train(
    data="dataset_obb_640/data.yaml",
    epochs=5, # quick test run, change to train real model
    imgsz=640,
    batch=16, # fit valuse to your hardware
    workers=8, # fit valuse to your hardware
    device=0,
    patience=25,
    optimizer="auto",
    project="runs",
    name="name_number_obb_model"
)
```

## Mobile Export (LiteRT / TFLite)

To run these models on Android with high frame rates using Google's LiteRT (formerly TFLite), export the trained PyTorch weights (**best.pt**, linux only):

```bash
# Export Pose Model to Float32 LiteRT
yolo export model="runs/card_pose_model/weights/best.pt" format=litert quantize=fp32

# Export with INT8 Quantization (for edge NPU acceleration)
yolo export model="runs/card_pose_model/weights/best.pt" format=litert quantize=8 data="data.yaml"
```

## Pre-trained Weights

Pre-trained weights (.pt and .tflite formats) for both card corner detection and name/number localization are available under the [models](./models/).

The text recognition step in Price Corner uses PP-OCRv6 tiny, used unmodyfied from [huggingface](https://huggingface.co/PaddlePaddle/PP-OCRv6_tiny_rec_onnx).


## Credits

- Card detection and name/number localization models were trained using [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) (pose and OBB architectures)
- Text recognition uses [PP-OCRv6 tiny](https://huggingface.co/PaddlePaddle/PP-OCRv6_tiny_rec_onnx) from the [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) project, used unmodified.
- Card data from [TCGdex](https://tcgdex.dev).

## License

Licensed under the Apache License, Version 2.0 - see [LICENSE](./LICENSE) for the full text.

## Disclaimer

Price Corner is an independent, fan-made tool and is not affiliated with, endorsed by, or sponsored by Nintendo, The Pokémon Company, Cardmarket, TCGPlayer, PriceCharting, TCGCollector, or TCGdex. All trademarks belong to their respective owners.

## Support 

You can support me by staring the repo or by sending me a [donation](https://ko-fi.com/oftenfive) - all voluntary.