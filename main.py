import cv2
import mediapipe as mp

# Initialize MediaPipe
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

# Initialize Hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

# Define finger tips
fingers_tips = [4, 8, 12, 16, 20]

# Webcam
cap = cv2.VideoCapture(0)

def fingers_up(lm_list):
    fingers = []

    # Thumb
    if lm_list[4][0] > lm_list[3][0]:  # For right hand
        fingers.append(1)
    else:
        fingers.append(0)

    # 4 Fingers
    for tip in fingers_tips[1:]:
        if lm_list[tip][1] < lm_list[tip - 2][1]:
            fingers.append(1)
        else:
            fingers.append(0)
    return fingers

def get_gesture(fingers):
    if fingers == [0, 0, 0, 0, 0]:
        return "Fist"
    elif fingers == [1, 1, 0, 0, 0]:
        return "Peace"
    elif fingers == [1, 0, 0, 0, 0]:
        return "Thumbs Up"
    elif fingers == [1, 1, 1, 1, 1]:
        return "Open Hand"
    else:
        return "Unknown"

while True:
    success, img = cap.read()
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hands.process(img_rgb)

    lm_list = []

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            for id, lm in enumerate(hand_landmarks.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lm_list.append((cx, cy))

            mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        if lm_list:
            finger_state = fingers_up(lm_list)
            gesture = get_gesture(finger_state)
            cv2.putText(img, gesture, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

    cv2.imshow("Hand Sign Detection", img)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
