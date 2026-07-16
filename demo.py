
import json
import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model

# Load trained models
emotion_model = load_model('emotion_model_best.h5')
drowsiness_model = load_model('drowsiness_model_best.h5')

# Load label order exactly as produced during training, instead of
# hardcoding/assuming alphabetical order.
with open('emotion_class_indices.json') as f:
    _emotion_indices = json.load(f)
EMOTION_LABELS = [None] * len(_emotion_indices)
for name, idx in _emotion_indices.items():
    EMOTION_LABELS[idx] = name

with open('eye_class_indices.json') as f:
    _eye_indices = json.load(f)
EYE_LABELS = [None] * len(_eye_indices)
for name, idx in _eye_indices.items():
    EYE_LABELS[idx] = name

HIGH_RISK_EMOTIONS = ['Angry', 'Fear', 'Sad']

# Face detector (bounding box) + Face Mesh (landmarks, for accurate eye crop)
mp_face = mp.solutions.face_detection
mp_mesh = mp.solutions.face_mesh
face_detector = mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.6)
face_mesh = mp_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True,
                              min_detection_confidence=0.6, min_tracking_confidence=0.6)

# MediaPipe Face Mesh landmark indices around each eye
LEFT_EYE_IDX = [33, 133, 159, 145, 160, 144, 163, 7]
RIGHT_EYE_IDX = [362, 263, 386, 374, 387, 373, 390, 249]


def preprocess_face(frame, bbox, size=96):
    x, y, w, h = bbox
    face = frame[max(0, y):y + h, max(0, x):x + w]
    if face.size == 0:
        return None
    face = cv2.resize(face, (size, size))
    face = face.astype('float32') / 255.0
    return np.expand_dims(face, axis=0)


def get_eye_crop(frame, landmarks, idx_list, w_frame, h_frame, pad=8, size=96):
    xs = [int(landmarks[i].x * w_frame) for i in idx_list]
    ys = [int(landmarks[i].y * h_frame) for i in idx_list]
    x1, x2 = max(0, min(xs) - pad), min(w_frame, max(xs) + pad)
    y1, y2 = max(0, min(ys) - pad), min(h_frame, max(ys) + pad)
    if x2 <= x1 or y2 <= y1:
        return None
    eye = frame[y1:y2, x1:x2]
    if eye.size == 0:
        return None
    eye = cv2.resize(eye, (size, size))
    eye = eye.astype('float32') / 255.0
    return np.expand_dims(eye, axis=0)


# Webcam initialisation
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print('Inference running. Press Q to exit.')

frame_count = 0
current_emotion = 'Detecting...'
current_eye_state = 'Detecting...'

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    display = frame.copy()
    h_frame, w_frame = frame.shape[:2]

    # Inference every 3 frames to keep the loop responsive
    if frame_count % 3 == 0:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        det_results = face_detector.process(rgb)

        if det_results.detections:
            detection = det_results.detections[0]
            bbox_norm = detection.location_data.relative_bounding_box

            x = max(0, int(bbox_norm.xmin * w_frame))
            y = max(0, int(bbox_norm.ymin * h_frame))
            w = int(bbox_norm.width * w_frame)
            h = int(bbox_norm.height * h_frame)

            cv2.rectangle(display, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Emotion prediction on the full face crop
            face_input = preprocess_face(frame, (x, y, w, h))
            if face_input is not None:
                probs = emotion_model.predict(face_input, verbose=0)[0]
                current_emotion = EMOTION_LABELS[int(np.argmax(probs))]

            # Eye-state prediction using landmark-based crops (both eyes)
            mesh_results = face_mesh.process(rgb)
            if mesh_results.multi_face_landmarks:
                landmarks = mesh_results.multi_face_landmarks[0].landmark
                left_input = get_eye_crop(frame, landmarks, LEFT_EYE_IDX, w_frame, h_frame)
                right_input = get_eye_crop(frame, landmarks, RIGHT_EYE_IDX, w_frame, h_frame)

                eye_preds = []
                for eye_input in (left_input, right_input):
                    if eye_input is not None:
                        p = drowsiness_model.predict(eye_input, verbose=0)[0][0]
                        eye_preds.append(p)

                if eye_preds:
                    avg_pred = float(np.mean(eye_preds))
                    current_eye_state = EYE_LABELS[int(avg_pred > 0.5)]

    # Overlay results
    risk_flag = current_emotion in HIGH_RISK_EMOTIONS or current_eye_state.lower().startswith('close')
    color = (0, 0, 255) if risk_flag else (0, 200, 0)

    cv2.putText(display, f'Emotion: {current_emotion}', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    cv2.putText(display, f'Eyes: {current_eye_state}', (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    if risk_flag:
        cv2.putText(display, 'RISK DETECTED', (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    cv2.imshow('Driver Safety Monitor', display)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
