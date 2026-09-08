# Persian Plate Detector

Detects vehicles and reads Iranian license plates from images, using YOLOv8, EasyOCR, and Django.

## Features

- Vehicle detection (car, motorcycle, bus, truck) via YOLOv8
- Iranian plate localization with a dedicated plate model
- Plate text recognition via EasyOCR, parsed into the standard Iranian format (2 digits + letter + 3 digits + 2 digits)
- Web UI with detection history

## How it works

1. Detect vehicles in the image (YOLOv8)
2. Locate the plate on each vehicle (`plateYolo.pt`, falling back to image processing if unavailable)
3. Read the plate text (EasyOCR)
4. Parse the text into the Iranian plate format
5. Return the annotated image with results

## Installation

```bash
git clone https://github.com/Amirmn24/persian-plate-detector.git
cd persian-plate-detector
pip install -r requirements.txt

python manage.py migrate
python manage.py runserver
```

Open `http://127.0.0.1:8000`.

### Models

The YOLOv8 vehicle model downloads automatically on first run. For the plate detection model:

```bash
python download_plate_model.py
```

If that fails, download `plateYolo.pt` manually from [persian-license-plate-recognition](https://github.com/truthofmatthew/persian-license-plate-recognition/tree/main/model) and place it in `model/`. For better character extraction, add `CharsYolo.pt` from the same repo — otherwise plate text falls back to EasyOCR.

## Project structure

```
persian-plate-detector/
├── detector/     # Django app (models, views, forms)
├── utlis/        # vehicle detection & plate reading
├── templates/    # HTML templates
├── config/       # Django settings
├── models/       # YOLO model files
└── manage.py
```

## Tech stack

Django · YOLOv8 (Ultralytics) · EasyOCR · OpenCV · Bootstrap 5 · SQLite

## License

MIT