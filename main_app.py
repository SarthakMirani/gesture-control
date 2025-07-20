import cv2
import mediapipe as mp

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
last_gesture = None  # Remember the last detected gesture

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    frame = cv2.flip(frame, 1)  # Flip image for natural interaction
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    # Handedness info (Left/Right)
    handedness_label = None
    if results.multi_handedness:
        for hand_handedness in results.multi_handedness:
            handedness_label = hand_handedness.classification[0].label  # 'Left' or 'Right'

    gesture = "Unknown"
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Convert landmarks to pixel coordinates
            landmarks = []
            h, w, _ = frame.shape
            for lm in hand_landmarks.landmark:
                cx, cy = int(lm.x * w), int(lm.y * h)
                landmarks.append((cx, cy))

            # Define tip and base indices
            finger_tips = [8, 12, 16, 20]
            finger_bases = [6, 10, 14, 18]

            # Count folded fingers (excluding thumb)
            fingers_folded = sum(
                landmarks[tip][1] > landmarks[base][1]
                for tip, base in zip(finger_tips, finger_bases)
            )

            # Thumb-specific detection
            thumb_tip = landmarks[4]
            thumb_ip = landmarks[3]
            thumb_cmc = landmarks[2]

            thumb_extended_horizontally = (
                (handedness_label == "Right" and thumb_tip[0] > thumb_ip[0] > thumb_cmc[0]) or
                (handedness_label == "Left" and thumb_tip[0] < thumb_ip[0] < thumb_cmc[0])
            )
            thumb_up_vertically = thumb_tip[1] < thumb_ip[1] - 20  # buffer for stability

            # Gesture logic
            if (
                thumb_extended_horizontally and
                thumb_up_vertically and
                all(landmarks[tip][1] > landmarks[base][1] for tip, base in zip(finger_tips, finger_bases))
            ):
                gesture = "Thumbs Up"
            elif (
                landmarks[8][1] < landmarks[6][1] and
                landmarks[12][1] < landmarks[10][1] and
                landmarks[16][1] > landmarks[14][1] and
                landmarks[20][1] > landmarks[18][1]
            ):
                gesture = "Peace"
            elif fingers_folded == 4:
                gesture = "Fist"
            elif fingers_folded == 0:
                gesture = "Open Hand"

            break  # Only process one hand

    # Display gesture if changed
    if gesture != last_gesture:
        print(f"Gesture: {gesture} → Action: [Pending mapping]")
        last_gesture = gesture

    # Draw gesture label
    cv2.putText(frame, f"Gesture: {gesture}", (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # Exit message
    cv2.putText(frame, "Press ESC to exit", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 4)
    cv2.putText(frame, "Press ESC to exit", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Hand Tracker", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
