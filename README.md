# Mediapipe hand tracking web-server

## Instructions

### Recommended to use venv

```
virtualenv .venv
source .venv/bin/activate
```

### Install dependencies

```
pip install -r requirements
```

### Download mediapipe model and put it in folder

For more details: [MediaPipe](https://ai.google.dev/edge/mediapipe/solutions/vision/gesture_recognizer)

```
wget -q https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/latest/gesture_recognizer.task
```

### Run

```
# running in interactive mode:

python app.py --resolution-width 1920 --resolution-height 1080 --camera 0

# running in server mode (no window):

python app.py --server --port 8765
```
