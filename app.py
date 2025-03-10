#@markdown We implemented some functions to visualize the hand landmark detection results. <br/> Run the following cell to activate the functions.

import mediapipe as mp
from mediapipe import solutions
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import cv2
import os
import threading
import asyncio
import websockets
import orjson
import argparse

LANDMARKS = [
    "WRIST",
    "THUMB_CMC",
    "THUMB_MCP",
    "THUMB_IP",
    "THUMB_TIP",
    "INDEX_FINGER_MCP",
    "INDEX_FINGER_PIP",
    "INDEX_FINGER_DIP",
    "INDEX_FINGER_TIP",
    "MIDDLE_FINGER_MCP",
    "MIDDLE_FINGER_PIP",
    "MIDDLE_FINGER_DIP",
    "MIDDLE_FINGER_TIP",
    "RING_FINGER_MCP",
    "RING_FINGER_PIP",
    "RING_FINGER_DIP",
    "RING_FINGER_TIP",
    "PINKY_MCP",
    "PINKY_PIP",
    "PINKY_DIP",
    "PINKY_TIP"
]

MARGIN = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54) # vibrant green

# Shared global state
clients = set()
lastResults = None
lock = threading.Lock()  # Thread lock to avoid race conditions

def draw_landmarks_on_image(rgb_image, detection_result):
  gestures = detection_result.gestures
  hand_landmarks_list = detection_result.hand_landmarks
  handedness_list = detection_result.handedness

  if len(gestures) == 0:
      return rgb_image

  annotated_image = np.copy(rgb_image)

  # Loop through the detected hands to visualize.
  for idx in range(len(hand_landmarks_list)):
    hand_landmarks = hand_landmarks_list[idx]
    handedness = handedness_list[idx]
    try:
        gesture = gestures[idx][0]
    except TypeError:
        gesture = dict()
        gesture['category_name'] = "Unknown"
        gesture['score'] = 1

    # Draw the hand landmarks.
    hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    hand_landmarks_proto.landmark.extend([
      landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks
    ])
    solutions.drawing_utils.draw_landmarks(
      annotated_image,
      hand_landmarks_proto,
      solutions.hands.HAND_CONNECTIONS,
      solutions.drawing_styles.get_default_hand_landmarks_style(),
      solutions.drawing_styles.get_default_hand_connections_style())

    # Get the top left corner of the detected hand's bounding box.
    height, width, _ = annotated_image.shape
    x_coordinates = [landmark.x for landmark in hand_landmarks]
    y_coordinates = [landmark.y for landmark in hand_landmarks]
    text_x = int(min(x_coordinates) * width)
    text_y = int(min(y_coordinates) * height) - MARGIN

    title = f"{gesture.category_name} ({gesture.score:.2f})"
    # Draw handedness (left or right hand) on the image.
    cv2.putText(annotated_image, title,
                (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX,
                FONT_SIZE, HANDEDNESS_TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)

  return annotated_image


BaseOptions = mp.tasks.BaseOptions
GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
GestureRecognizerResult = mp.tasks.vision.GestureRecognizerResult
VisionRunningMode = mp.tasks.vision.RunningMode

# Create a hand landmarker instance with the live stream mode:
def print_result(result: GestureRecognizerResult, output_image: mp.Image, timestamp_ms: int):
    # print('hand landmarker result: {}'.format(result))
    global lastResults
    with lock:
        lastResults = result

def recognize_gestures(src, width, height, isRunningInServerMode):
    if not cv2.cuda.getCudaEnabledDeviceCount():
        print("CUDA is not available.")
        use_cuda = False
    else:
        use_cuda = True
        print("CUDA is available.")

    # Use multi-threaded capture
    cap = cv2.VideoCapture(src)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
    cap.set(cv2.CAP_PROP_FPS, 30)

    # Check if the webcam is opened correctly
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        exit()

    options = GestureRecognizerOptions(
        base_options=BaseOptions(model_asset_path='./gesture_recognizer.task'),
        running_mode=VisionRunningMode.LIVE_STREAM,
        num_hands=2,
        result_callback=print_result)
    with GestureRecognizer.create_from_options(options) as landmarker:
        while True:
            ret, frame = cap.read()  # Read the latest frame
            if not ret:
                print("Error: Couldn't read frame.")
                break

            if use_cuda:
                gpu_image = cv2.cuda_GpuMat()
                gpu_image.upload(frame)

                # Flip the image on the GPU
                gpu_flipped = cv2.cuda.flip(gpu_image, 1)  # Flip horizontally

                # Download back to CPU
                frame = gpu_flipped.download()
            else:
                frame = cv2.flip(frame, 1)

            frame_timestamp_ms = int(cv2.getTickCount() / cv2.getTickFrequency() * 1000) 
            if use_cuda:
                # Upload frame to GPU
                gpu_frame = cv2.cuda_GpuMat()
                gpu_frame.upload(frame)

                # Resize on GPU
                gpu_resized = cv2.cuda.resize(gpu_frame, (640, 480))  # Resize to 640x360

                # Download back to CPU
                frame_small = gpu_resized.download()
            else:
                # CPU fallback
                frame_small = cv2.resize(frame, (640, 480))

            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
            landmarker.recognize_async(mp_image, frame_timestamp_ms)
            if lastResults is not None:
                frame = draw_landmarks_on_image(frame, lastResults)

            if not isRunningInServerMode:
                cv2.imshow("Webcam Feed", frame)  # Display the frame

            if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit
                break

        stream.stop()
        cv2.destroyAllWindows()

def serialize_results():
    global lastResults
    if lastResults is None:
        return None

    resultDict = dict()

    serializedGestures = []
    for idx, gestures in enumerate(lastResults.gestures):
        gs = [{"category": g.category_name, "score": g.score} for g in gestures]
        serializedGestures.append(gs)

    serializedHandLandmarks = [ ]
    for idx, hand_landmarks in enumerate(lastResults.hand_landmarks):
        hs = [{"x": l.x, "y": l.y, "z": l.z, "type": LANDMARKS[idp]} for idp, l in enumerate(hand_landmarks)]
        serializedHandLandmarks.append(hs)

    serializedHandedness = [ ]
    for idx, handedness in enumerate(lastResults.handedness):
        hh = [{"hand": h.category_name, "score": h.score} for h in handedness]
        serializedHandedness.append(hh)
                               
    # world_landmarks_list = lastResults.world_landmarks

    resultDict["gestures"] = serializedGestures
    resultDict["hand_landmarks"] = serializedHandLandmarks
    resultDict["handedness"] = serializedHandedness

    return orjson.dumps(resultDict).decode("utf-8")


async def websocket_handler(websocket):
    """Handle WebSocket connections."""
    clients.add(websocket)
    try:
        while True:
            await websocket.send(serialize_results())  # Send latest hand landmarks
            await asyncio.sleep(0.01)  # Prevents busy-waiting
    except websockets.exceptions.ConnectionClosed:
        pass
    finally:
        clients.remove(websocket)

async def start_server(port):
    """Start the WebSocket server."""
    async with websockets.serve(websocket_handler, "127.0.0.1", port):
        print(f"WebSocket server started on ws://localhost:{port}")
        await asyncio.Future()  # Keeps the server running

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gesture recognition server")
    parser.add_argument("--server", type=bool, default=False ,help="Run in server mode. Will not draw output.")
    parser.add_argument("--port", type=int, default=8765 ,help="Websocket port. Default is 8765")
    parser.add_argument("--camera", type=int, default=0, help="The id of the webcam - Default is 0 and it grabs the first one")
    parser.add_argument("--resolution-width", type=int, default=1920, help="Width of webcam resolution - in pixels. Default is 1920")
    parser.add_argument("--resolution-height", type=int, default=1080, help="Height of webcam resolution - in pixels. Default is 1080")
    args = parser.parse_args()

    # Start video capture in a separate thread
    video_thread = threading.Thread(target=recognize_gestures, args=(args.camera, args.resolution_width, args.resolution_height, args.server), daemon=True)
    video_thread.start()

    # Start WebSocket server in the main thread
    asyncio.run(start_server(args.port))
