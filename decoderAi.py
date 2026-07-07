# import cv2
# import numpy as np
# import math
# import mediapipe as mp
# import pandas as pd
# import joblib
# import csv

# baseOptions = mp.tasks.BaseOptions
# hand_landmarker = mp.tasks.vision.HandLandmarker
# hand_landmarker_options = mp.tasks.vision.HandLandmarkerOptions
# vision_running_mode = mp.tasks.vision.RunningMode


# options = hand_landmarker_options(base_options = baseOptions(model_asset_path="/Users/vaishnavravulaparthi/Python_Projects/ASL/hand_landmarker.task"), num_hands=1, min_hand_detection_confidence = 0.8, running_mode=vision_running_mode.IMAGE)

# landmarker = hand_landmarker.create_from_options(options)

# camera = cv2.VideoCapture(0)

# conn = mp.tasks.vision.HandLandmarksConnections.HAND_CONNECTIONS

# model = joblib.load("asl_model.pkl")
# encoder = joblib.load("label_encoder.pkl")



# def normalizer(hand_coordinates):

#     centered = hand_coordinates - hand_coordinates[0]

#     # 1. Define 'i' (Normalized Wrist to Middle MCP)
#     v_i = centered[9]
#     i = v_i / np.linalg.norm(v_i)
    
#     # 2. Define a raw horizontal vector 'v_j' (Wrist to Pinky MCP)
#     v_j_raw = centered[17]
    
#     # 3. Use Gram-Schmidt to make 'j' perpendicular to 'i'
#     # j = v_j_raw - projection of v_j_raw onto i
#     projection = np.dot(v_j_raw, i) * i
#     v_j_ortho = v_j_raw - projection
#     j = v_j_ortho / np.linalg.norm(v_j_ortho)
    
#     # 4. Find 'k' (The Normal vector) using the Cross Product
#     # k is perpendicular to both i and j
#     k = np.cross(i, j)

#     p = np.stack([i, j, k]) # transpose this
#     p_inverse = p.T

#     # Find inverse of p

#     new_coords = centered @ p_inverse

#     len = np.linalg.norm(v_i)
#     normalized_coords = new_coords / len

#     return normalized_coords, p


# frame_counter = 0
# f_counter = 0

# fmv = []

# f_m_v = []

# previous_pinky = None

# while True:
#     success, frame = camera.read()
#     if not success:
#         break

#     frame = cv2.flip(frame, 1)

#     rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

#     h, w, c = rgb_frame.shape

#     mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

#     results = landmarker.detect(mp_image)
    
#     if results.hand_landmarks:

#         hand_coordinates = []
#         hand_2D_coordinates = []

#         for hand in results.hand_landmarks:

#             for point in hand:
#                 x = int((point.x)*w)
#                 y = int((point.y)*h)
#                 z = point.z
#                 hand_coordinates.append((x,y,z))
#                 hand_2D_coordinates.append((x,y))

#         for point in hand_2D_coordinates:
#             cv2.circle(frame, point, 5, (0, 255, 0), -1)
        
#         lines = []

#         for i in range(0,len(conn)):
#             cv2.line(frame, hand_2D_coordinates[conn[i].start], hand_2D_coordinates[conn[i].end], (255, 0, 0), 2)
#             lines.append([hand_2D_coordinates[conn[i].start], hand_2D_coordinates[conn[i].end]])

#         hand_coordinates = np.array(hand_coordinates)

#         normalized_coordinates, p = normalizer(hand_coordinates)

#         magnitudes = np.linalg.norm(normalized_coordinates, axis=1)

#         feature_vector = np.concatenate([normalized_coordinates.flatten(), magnitudes])

#         # if keyboard strike letter, then save the current feature vector to the associated letter
#         key = cv2.waitKey(1) & 0xFF
#         if key >= ord("a") and key <= ord("z"):
#             label = chr(key).upper()
#             with open("dataset.csv", "a", newline="") as f:
#                 writer = csv.writer(f)
#                 writer.writerow([label]+feature_vector.tolist())

#             print(f"Saved {label}")

#         prediction = model.predict([feature_vector])

#         letter = encoder.inverse_transform(prediction)

#         cv2.putText(frame, f"Fingers: {letter[0]}", (300, 300),cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)

#     cv2.imshow("ASL Decoder Feed", frame)
#     if cv2.waitKey(1) & 0xFF == ord('.'):
#         break

# camera.release()
# cv2.destroyAllWindows()

"""
Real-Time American Sign Language (ASL) Decoder

This program:
1. Detects hand landmarks using MediaPipe.
2. Normalizes hand coordinates into a rotation-invariant coordinate system.
3. Extracts spatial features from hand landmarks.
4. Uses a trained machine learning model to classify ASL letters.
5. Allows additional training data collection through keyboard input.

Author: Vaishnav Ravulaparthi
"""

import cv2
import numpy as np
import mediapipe as mp
import joblib
import csv


# ============================================================
# MediaPipe Hand Landmarker Configuration
# ============================================================

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode


MODEL_PATH = (
    "/Users/vaishnavravulaparthi/Python_Projects/ASL/"
    "hand_landmarker.task"
)

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    num_hands=1,
    min_hand_detection_confidence=0.8,
    running_mode=RunningMode.IMAGE
)

landmarker = HandLandmarker.create_from_options(options)


# ============================================================
# Load Machine Learning Model
# ============================================================

try:
    model = joblib.load("asl_model.pkl")
    encoder = joblib.load("label_encoder.pkl")
except Exception as error:
    raise Exception(f"Unable to load ML model files: {error}")


# ============================================================
# Camera Initialization
# ============================================================

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    raise Exception("Unable to access camera")


connections = mp.tasks.vision.HandLandmarksConnections.HAND_CONNECTIONS


# ============================================================
# Coordinate Normalization
# ============================================================

def normalize_hand(hand_coordinates):
    """
    Converts raw hand landmark coordinates into a normalized
    coordinate system independent of hand position and rotation.

    The coordinate system is defined using:
    - i-axis: wrist to middle finger MCP
    - j-axis: perpendicular horizontal hand direction
    - k-axis: normal vector from cross product

    Parameters:
        hand_coordinates (np.ndarray):
            Array containing 21 MediaPipe hand landmarks.

    Returns:
        normalized_coordinates:
            Rotation and scale normalized landmarks.

        transformation_matrix:
            Coordinate basis matrix.
    """

    # Move wrist landmark to origin
    centered = hand_coordinates - hand_coordinates[0]


    # --------------------------------------------------------
    # Define i-axis using wrist -> middle MCP joint
    # --------------------------------------------------------

    wrist_to_middle = centered[9]

    i_axis = wrist_to_middle / np.linalg.norm(wrist_to_middle)


    # --------------------------------------------------------
    # Define j-axis using wrist -> pinky MCP
    # Orthogonalize using Gram-Schmidt
    # --------------------------------------------------------

    wrist_to_pinky = centered[17]

    projection = np.dot(wrist_to_pinky, i_axis) * i_axis

    j_axis_raw = wrist_to_pinky - projection

    j_axis = j_axis_raw / np.linalg.norm(j_axis_raw)


    # --------------------------------------------------------
    # Define k-axis perpendicular to hand plane
    # --------------------------------------------------------

    k_axis = np.cross(i_axis, j_axis)


    coordinate_system = np.stack(
        [i_axis, j_axis, k_axis]
    )


    # Transform coordinates into new reference frame
    transformed_coordinates = centered @ coordinate_system.T


    # Normalize scale using wrist-middle distance
    hand_scale = np.linalg.norm(wrist_to_middle)

    normalized_coordinates = (
        transformed_coordinates / hand_scale
    )


    return normalized_coordinates, coordinate_system



# ============================================================
# Real-Time ASL Recognition Loop
# ============================================================

while True:

    success, frame = camera.read()

    if not success:
        break


    # Mirror camera feed
    frame = cv2.flip(frame, 1)


    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    height, width, _ = rgb_frame.shape


    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )


    results = landmarker.detect(mp_image)


    if results.hand_landmarks:


        hand_coordinates = []
        hand_2D_coordinates = []


        # ----------------------------------------------------
        # Extract MediaPipe landmarks
        # ----------------------------------------------------

        for hand in results.hand_landmarks:

            for point in hand:

                x = int(point.x * width)
                y = int(point.y * height)

                hand_coordinates.append(
                    (x, y, point.z)
                )

                hand_2D_coordinates.append(
                    (x, y)
                )


        # ----------------------------------------------------
        # Draw hand landmarks
        # ----------------------------------------------------

        for point in hand_2D_coordinates:
            cv2.circle(
                frame,
                point,
                5,
                (0, 255, 0),
                -1
            )


        for connection in connections:

            cv2.line(
                frame,
                hand_2D_coordinates[connection.start],
                hand_2D_coordinates[connection.end],
                (255, 0, 0),
                2
            )


        hand_coordinates = np.array(
            hand_coordinates
        )


        # ----------------------------------------------------
        # Feature Extraction Pipeline
        # ----------------------------------------------------

        normalized_coordinates, _ = normalize_hand(
            hand_coordinates
        )


        landmark_distances = np.linalg.norm(
            normalized_coordinates,
            axis=1
        )


        feature_vector = np.concatenate(
            [
                normalized_coordinates.flatten(),
                landmark_distances
            ]
        )


        # ----------------------------------------------------
        # Dataset Collection Mode
        #
        # Press any alphabet key to save the current hand
        # position as a training example.
        # ----------------------------------------------------

        key = cv2.waitKey(1) & 0xFF

        if ord("a") <= key <= ord("z"):

            label = chr(key).upper()

            with open(
                "dataset.csv",
                "a",
                newline=""
            ) as file:

                writer = csv.writer(file)

                writer.writerow(
                    [label] + feature_vector.tolist()
                )


            print(f"Saved training sample: {label}")


        # ----------------------------------------------------
        # Machine Learning Prediction
        # ----------------------------------------------------

        prediction = model.predict(
            [feature_vector]
        )

        letter = encoder.inverse_transform(
            prediction
        )


        cv2.putText(
            frame,
            f"Prediction: {letter[0]}",
            (50, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            2,
            (0, 255, 0),
            3
        )


    cv2.imshow(
        "ASL Decoder Feed",
        frame
    )


    # Press '.' to exit
    if cv2.waitKey(1) & 0xFF == ord("."):
        break



# ============================================================
# Cleanup
# ============================================================

camera.release()
cv2.destroyAllWindows()