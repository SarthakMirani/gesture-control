import cv2
import mediapipe as mp
import subprocess

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7
)
mp_draw = mp.solutions.drawing_utils

# Open webcam
cap = cv2.VideoCapture(0)

gesture_triggered = False

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    # Flip image for natural interaction
    frame = cv2.flip(frame, 1)

    # Convert BGR to RGB
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    # If a hand is detected
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
            )

            # Convert landmarks to pixel coordinates
            landmark_list = []
            h, w, _ = frame.shape
            for id, lm in enumerate(hand_landmarks.landmark):
                cx, cy = int(lm.x * w), int(lm.y * h)
                landmark_list.append((cx, cy))

            # Gesture detection (fist vs open hand)
            if landmark_list:
                fingers_folded = 0
                finger_tips = [8, 12, 16, 20]  # Tips of index to pinky
                finger_bases = [6, 10, 14, 18]  # Corresponding base joints

                for tip, base in zip(finger_tips, finger_bases):
                    if landmark_list[tip][1] > landmark_list[base][1]:
                        fingers_folded += 1

                if fingers_folded == 4:
                    gesture = "Fist"
                    if not gesture_triggered:
                        subprocess.run(["systemctl", "poweroff"])
                        gesture_triggered = True
                else:
                    gesture = "Open Hand"
                    gesture_triggered = False  # Reset when hand is open

                # Draw gesture label
                cv2.putText(
                    frame,
                    f"Gesture: {gesture}",
                    (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 0, 0),
                    2
                )

    # Draw exit message with black border
    cv2.putText(frame, "Press ESC to exit", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 4)
    cv2.putText(frame, "Press ESC to exit", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow("Hand Tracker", frame)

    # Exit on ESC key
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
