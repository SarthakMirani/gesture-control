import cv2
import mediapipe as mp

# Initialize MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Define gesture-action mappings
gesture_actions = {
    "Fist": "Shutdown",
    "Open Hand": "Open Browser",
    "Peace": "Lock Screen"
}

# Open webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        continue

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    gesture = None

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Convert landmarks to pixels
            h, w, _ = frame.shape
            landmarks = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks.landmark]

            # Identify finger states
            tips = [8, 12, 16, 20]
            bases = [6, 10, 14, 18]

            fingers_folded = sum(landmarks[tip][1] > landmarks[base][1] for tip, base in zip(tips, bases))

            # Gesture logic
            if fingers_folded == 4:
                gesture = "Fist"
            elif fingers_folded == 0:
                gesture = "Open Hand"
            elif (
                landmarks[8][1] < landmarks[6][1] and  # index up
                landmarks[12][1] < landmarks[10][1] and  # middle up
                landmarks[16][1] > landmarks[14][1] and  # ring down
                landmarks[20][1] > landmarks[18][1]      # pinky down
            ):
                gesture = "Peace"

            # Show gesture → action
            if gesture:
                action = gesture_actions.get(gesture, "Unknown")
                print(f"Gesture: {gesture} → Action: {action}")

                # Draw on screen
                cv2.putText(frame, f"{gesture} → {action}", (10, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    # Instructions
    cv2.putText(frame, "Press ESC to exit", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 4)
    cv2.putText(frame, "Press ESC to exit", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Gesture Mapper", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
