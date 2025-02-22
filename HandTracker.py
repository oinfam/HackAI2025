# STEP 1: Import the necessary modules.
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
from mediapipe.tasks.python import vision
import cv2
import math
import mouse
import time

MARGIN = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54) # vibrant green



def draw_landmarks_on_image(rgb_image, detection_result):
  hand_landmarks_list = detection_result.hand_landmarks
  handedness_list = detection_result.handedness
  top_gesture = detection_result.gestures
  if np.size(handedness_list) == 0:
    print("No Hand Detected")
    detected_gesture = "No Hand"
    indexx = 0
    indexy = 0
  else:
    detected_gesture = top_gesture[0][0].category_name
    indexx = hand_landmarks_list[0][8].x
    indexy = hand_landmarks_list[0][8].y
  annotated_image = np.copy(rgb_image)

  """# Loop through the detected hands to visualize.
  for idx in range(len(hand_landmarks_list)):
    hand_landmarks = hand_landmarks_list[idx]
    handedness = handedness_list[idx]

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

    # Draw handedness (left or right hand) on the image.
    cv2.putText(annotated_image, f"{handedness[0].category_name}",
                (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX,
                FONT_SIZE, HANDEDNESS_TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)"""

  return(1, detected_gesture, indexx, indexy)
  #return (annotated_image, detected_gesture,indexx,indexy)


# STEP 2: Create an HandLandmarker object.
base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(base_options=base_options,
                                       num_hands=2)
detector = vision.HandLandmarker.create_from_options(options)

base_options_gestures = python.BaseOptions(model_asset_path='gesture_recognizer.task')
optionsGestures = vision.GestureRecognizerOptions(base_options=base_options_gestures)
recognizer = vision.GestureRecognizer.create_from_options(optionsGestures)

# STEP 4: Detect hand landmarks from the input image.
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb_frame = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)

    detection_result = recognizer.recognize(rgb_frame)
    
    #detection_result = detector.detect(rgb_frame)
    (annotated_image, detected_gesture, x, y) = draw_landmarks_on_image(rgb_frame.numpy_view(), detection_result)
    #cv2.imshow("hi", cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
   
   #Mouse Movement (it scared of you. stop it)
    if detected_gesture == "Pointing_Up":
      xmouse = math.floor((1-x) * 1536)
      ymouse = math.floor(y * 864)
      mouse.move(xmouse,ymouse, absolute = True)

   #Left Click
    if detected_gesture == "Open_Palm":
      mouse.click('left')
      time.sleep(1)
    
    
    if detected_gesture == "Thumb_Up":
      break

