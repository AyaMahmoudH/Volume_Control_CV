#%%
import cv2
import mediapipe as mp
import numpy as np
import math
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
#%%
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
#%%
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
#%%
volRange = volume.GetVolumeRange()
minVol = volRange[0]
maxVol = volRange[1]
#%%
cap = cv2.VideoCapture(0)
#%%
with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
    
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue
        
        image = cv2.flip(image, 1)
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        results = hands.process(image_rgb)
        
        image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS)
                
                landmarks = []
                for id, lm in enumerate(hand_landmarks.landmark):
                    h, w, c = image.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    landmarks.append([id, cx, cy])
                
                if landmarks:
                    x1, y1 = landmarks[4][1], landmarks[4][2]  
                    x2, y2 = landmarks[8][1], landmarks[8][2]  
                    
                    cv2.circle(image, (x1, y1), 10, (255, 0, 0), cv2.FILLED)
                    cv2.circle(image, (x2, y2), 10, (255, 0, 0), cv2.FILLED)
                    
                    cv2.line(image, (x1, y1), (x2, y2), (255, 0, 0), 3)
                    
                    length = math.hypot(x2 - x1, y2 - y1)
                    
                    vol = np.interp(length, [30, 250], [minVol, maxVol])
                    volume.SetMasterVolumeLevel(vol, None)
                    
                    vol_bar = np.interp(length, [30, 250], [400, 150])
                    cv2.rectangle(image, (50, 150), (85, 400), (0, 255, 0), 3)
                    cv2.rectangle(image, (50, int(vol_bar)), (85, 400), (0, 255, 0), cv2.FILLED)
                    
                    vol_per = np.interp(length, [30, 250], [0, 100])
                    cv2.putText(image, f'{int(vol_per)}%', (40, 450), 
                               cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 3)
        
        cv2.imshow('Volume Control', image)
        
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break
#%%
cap.release()
cv2.destroyAllWindows()