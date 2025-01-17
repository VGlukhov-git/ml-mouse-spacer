import cv2
import mediapipe as mp #0.10.20
import numpy as np
import pyautogui
import math

import tensorflow as tf #2.18.0

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

model = tf.keras.models.load_model('./my_model/model.keras', compile=False)

cap = cv2.VideoCapture(0)

pyautogui.PAUSE = 0

STATUS = -1
lastDistance = -1
lastCoordX = 0
lastCoordY = 0
isClicked = False

while cap.isOpened():
    success, image = cap.read()
    if not success:
        break
    
    image = cv2.flip(image, 1)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    results = hands.process(image_rgb)
    if results.multi_hand_world_landmarks:
        for world_landmark in results.multi_hand_world_landmarks:
            h, w, _ = image.shape
            hand_landmarks = results.multi_hand_landmarks[0]
            landmark = hand_landmarks.landmark

            pointA = (int(landmark[4].x * w), int(landmark[4].y * h))
            pointB = (int(landmark[8].x * w), int(landmark[8].y * h))
            pointC = (int(landmark[12].x * w), int(landmark[12].y * h))

            landmarks = world_landmark.landmark
            pose_array = np.array([[(lm.x), (lm.y), (lm.z)] for lm in landmarks])

            pose_input = pose_array
            values = pose_input.tolist()

            pose_input = np.expand_dims(pose_array, axis=0)  # Add batch dimension
            results = model.predict(pose_input)
            state = results.squeeze().argmax()
            prediction = results.squeeze()[state]

            if prediction > 0.95:
                # print(state, prediction)
                if  state == 0:
                    distance_AB = math.sqrt((pointA[0] - pointB[0])**2 + (pointA[1] - pointB[1])**2) / 10
                    if distance_AB < 8 and isClicked == False:
                        STATUS = 'CLICKED_DOWN'
                        screen_width, screen_height = pyautogui.size()
                        pyautogui.moveTo(screen_width/2, screen_height/2)
                        pyautogui.mouseDown(button='right')
                        pyautogui.keyDown('ctrl')
                        lastCoordX = pointA[0]
                        lastCoordY = pointA[1]
                        isClicked = True
                    if distance_AB > 12 and isClicked == True:
                        STATUS = 'CLICKED_UP'
                        pyautogui.keyUp('ctrl')
                        pyautogui.mouseUp(button='right')
                        lastCoordX = pointA[0]
                        lastCoordY = pointA[1]
                        isClicked = False
                    if distance_AB < 15 and abs(lastCoordX - pointA[0]) > 5 and abs(lastCoordY - pointA[1]) > 5:
                        dX = pointA[0] - lastCoordX
                        dY = pointA[1] - lastCoordY
                        if STATUS == 'CLICKED_DOWN':
                            mousex, mousey = pyautogui.position()
                            mousex += dX/1
                            mousey += dY/1
                            pyautogui.moveTo(mousex, mousey) #0.05
                        lastCoordX = pointA[0]
                        lastCoordY = pointA[1]
                if state == 1:
                    distance_AB = math.sqrt((pointA[0] - pointB[0])**2 + (pointA[1] - pointB[1])**2)
                    distance_BC = math.sqrt((pointB[0] - pointC[0])**2 + (pointB[1] - pointC[1])**2)
                    distance_CA = math.sqrt((pointC[0] - pointA[0])**2 + (pointC[1] - pointA[1])**2)

                    # Total distance
                    total_distance = distance_AB + distance_BC + distance_CA
                    if lastDistance == -1:
                        lastDistance = total_distance / 10
                    if lastDistance != -1:
                        deltaScroll = total_distance / 10 - lastDistance
                        lastDistance = total_distance / 10
                    if deltaScroll > 1 and deltaScroll < 30:
                        pyautogui.scroll(deltaScroll/5)
                    if deltaScroll < -1 and deltaScroll > -30:
                        pyautogui.scroll(deltaScroll/5)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
hands.close()