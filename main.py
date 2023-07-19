import cv2
import mediapipe as mp
from math import sqrt
import time 

video = cv2.VideoCapture(0)

hands = mp.solutions.hands
Hands = hands.Hands()
mpDwaw = mp.solutions.drawing_utils
beforex, beforey, beforez = 0.0,0.0,0.0
distancia_total, velocidade = 0.0,0.0
while True:
    success, img = video.read()
    img = cv2.flip(img,1)
    frameRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    results = Hands.process(frameRGB)
    handPoints = results.multi_hand_landmarks
    h, w, _ = img.shape
    tempo_inicial = time.time()
    pontos = []
    if handPoints:
        for points in handPoints:
            mpDwaw.draw_landmarks(img, points,hands.HAND_CONNECTIONS)
            for id, cord in enumerate(points.landmark):

                cx, cy = int(cord.x * w), int(cord.y * h)
                cv2.putText(img, str(id), (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                #cv2.circle(img,(cx,cy),15,(0,255,0),cv2.FILLED)
                if id == 0:
                    xp0 = cord.x
                    yp0 = cord.y
                    zp0 = cord.z
                elif id == 5:
                    xp1 = cord.x
                    yp1 = cord.y
                    zp1 = cord.z
                elif id == 17:
                    newpx = (xp1 + cord.x + xp0)/3
                    newpy = (yp1 + cord.y + yp0)/3
                    newpz = zp0
                    cv2.circle(img,(int(newpx *w), int(newpy*h)),4,(0,0,255),cv2.FILLED)

                    distancia = sqrt((newpx -beforex)**2 + (newpy-beforey)**2 + (newpz-beforez)**2)


                    tempo_decorrido = time.time() - tempo_inicial

                    beforex = newpx
                    beforey = newpy
                    beforez = newpz
                    if tempo_decorrido != 0:
                        velocidade = distancia/tempo_decorrido
                    cv2.putText(img,f"Velocidade: {velocidade:.2f} m/s",(0,50),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),5)

                pontos.append((cx,cy))
#cv2.circle(img,(cx,cy),15,(0,255,0),cv2.FILLED)


    cv2.imshow('Imagem',img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()

