"""
@author: Elena Eremia
@author: Floris Stoica
"""

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
import time

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
    global lastResults
    with lock:
        lastResults = result


# Check if CUDA is available
def check_cuda():
    if hasattr(cv2, 'cuda') and cv2.cuda.getCudaEnabledDeviceCount() > 0:
        print("CUDA is available.")
        return True
    print("CUDA is not available.")
    return False

# Check if OpenCL is available
def check_opencl():
    if hasattr(cv2, 'ocl') and cv2.ocl.haveOpenCL():
        print("OpenCL is available.")
        return True
    print("OpenCL is not available.")
    return False

# Test camera function to verify camera is working before main loop
def test_camera(src):
    print(f"Testing camera at index {src} with default settings...")
    test_cap = cv2.VideoCapture(src)
    if not test_cap.isOpened():
        print(f"Error: Could not open camera at index {src}")
        return False
    
    test_ret, test_frame = test_cap.read()
    test_cap.release()
    
    if not test_ret:
        print("Error: Could not read from camera with default settings")
        return False
    else:
        print("Successfully read a test frame with default settings")
        return True

# Function to recognize gestures with hardware acceleration when available
def recognize_gestures(src, width, height, isRunningInServerMode):
    # Check hardware acceleration availability
    use_cuda = check_cuda()
    use_opencl = check_opencl()
    
    acceleration_type = None
    if use_cuda:
        acceleration_type = "CUDA"
        print("Using CUDA for hardware acceleration.")
    elif use_opencl:
        acceleration_type = "OpenCL"
        print("Using OpenCL for hardware acceleration.")
    else:
        print("Falling back to CPU processing.")
    
    # Test the camera first
    if not test_camera(src):
        print("Camera test failed. Please check camera connections and permissions.")
        return
    
    # Initialize the camera with requested settings
    print(f"Attempting to open camera at index {src} with resolution {width}x{height}")
    cap = cv2.VideoCapture(src)
    
    # Set camera properties
    print(f"Setting resolution to {width}x{height}...")
    width_success = cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    height_success = cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    fps_success = cap.set(cv2.CAP_PROP_FPS, 30)
    
    # Verify actual camera settings
    actual_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    actual_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Actual camera settings: {actual_width}x{actual_height} @ {actual_fps}fps")
    
    # Test reading a frame with custom settings
    for attempt in range(5):
        ret, frame = cap.read()
        if ret:
            print(f"Successfully read frame on attempt {attempt+1}")
            break
        else:
            print(f"Failed to read frame on attempt {attempt+1}")
            time.sleep(1)
    
    if not ret:
        print("Error: Could not read frames with custom settings. Exiting.")
        cap.release()
        return
    
    # Create options for the gesture recognizer
    model_path = os.path.join(os.path.dirname(__file__), 'gesture_recognizer.task')
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        print("Please download the gesture_recognizer.task model from MediaPipe")
        cap.release()
        return
    
    options = GestureRecognizerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.LIVE_STREAM,
        num_hands=2,
        result_callback=print_result)
    
    # Main processing loop
    print("Starting main processing loop...")
    with GestureRecognizer.create_from_options(options) as recognizer:
        while True:
            ret, frame = cap.read()
            
            if not ret:
                print("Error: Failed to read frame in main loop")
                break
            
            # Process the frame with hardware acceleration if available
            if acceleration_type == "CUDA":
                # Upload to GPU
                gpu_frame = cv2.cuda_GpuMat()
                gpu_frame.upload(frame)
                
                # Process with CUDA
                gpu_flipped = cv2.cuda.flip(gpu_frame, 1)  # Horizontal flip
                frame = gpu_flipped.download()
            elif acceleration_type == "OpenCL":
                # Create UMat from frame
                ocl_frame = cv2.UMat(frame)
                # Process with OpenCL
                ocl_frame = cv2.flip(ocl_frame, 1)  # Horizontal flip
                frame = ocl_frame.get()
            else:
                # CPU processing
                frame = cv2.flip(frame, 1)
            
            # Convert the frame for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            

            
            # Process with the gesture recognizer
            frame_timestamp_ms = int(cv2.getTickCount() / cv2.getTickFrequency() * 1000)
            recognizer.recognize_async(mp_image, frame_timestamp_ms)
            
            # Resize the frame with hardware acceleration if available
            if acceleration_type == "CUDA":
                gpu_frame = cv2.cuda_GpuMat()
                gpu_frame.upload(frame)
                gpu_resized = cv2.cuda.resize(gpu_frame, (640, 480))
                frame = gpu_resized.download()
            elif acceleration_type == "OpenCL":
                ocl_frame = cv2.UMat(frame)
                ocl_frame = cv2.resize(ocl_frame, (640, 480))
                frame = ocl_frame.get()
            else:
                frame = cv2.resize(frame, (640, 480))
            
            # Show the annotated image if not in server mode
            if not isRunningInServerMode and lastResults is not None:
                with lock:
                    annotated_image = draw_landmarks_on_image(frame, lastResults)
                cv2.imshow("Gesture Recognition", annotated_image)
            elif not isRunningInServerMode:
                cv2.imshow("Webcam Feed", frame)
            
            # Check for exit key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    # Clean up
    cap.release()
    if not isRunningInServerMode:
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
        print("Client disconnected")
    finally:
        clients.remove(websocket)

async def start_server(port):
    """Start the WebSocket server."""
    async with websockets.serve(websocket_handler, "127.0.0.1", port):
        print(f"WebSocket server started on ws://localhost:{port}")
        await asyncio.Future()  # Keeps the server running

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gesture recognition server")
    parser.add_argument("--server", action="store_true", help="Run in server mode. Will not draw output.")
    parser.add_argument("--port", type=int, default=8765, help="Websocket port. Default is 8765")
    parser.add_argument("--camera", type=int, default=0, help="The id of the webcam - Default is 0 and it grabs the first one")
    parser.add_argument("--resolution-width", type=int, default=1920, help="Width of webcam resolution - in pixels. Default is 1920")
    parser.add_argument("--resolution-height", type=int, default=1080, help="Height of webcam resolution - in pixels. Default is 1080")
    parser.add_argument("--force-cpu", action="store_true", help="Force CPU processing even if hardware acceleration is available")
    args = parser.parse_args()

    # Start video capture in a separate thread
    video_thread = threading.Thread(target=recognize_gestures, args=(args.camera, args.resolution_width, args.resolution_height, args.server), daemon=True)
    video_thread.start()

    # Start WebSocket server in the main thread
    asyncio.run(start_server(args.port))