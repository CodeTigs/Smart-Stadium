import cv2
import mediapipe as mp
from math import sqrt,atan2, degrees
import time 
import numpy as np
#from mediapipe.python.solutions.hands import HandLandmark

class point:
    def __init__(self,*arg):
        if len(arg) == 3:
            self.x, self.y, self.z = arg[0],arg[1],arg[2]
        else:
            self.x, self.y, self.z = np.float32(arg[0].x), np.float32(arg[0].y), np.float32(arg[0].z)

    def calculoDistancia(self,point):
        return sqrt((self.x-point.x)**2 + (self.y-point.y)**2 + (self.z-point.z)**2)
    
    def calculoAngulo(self, point):
        dy = abs(point.y - self.y)
        dxz = sqrt((self.x - point.x) ** 2 + (self.z - point.z) ** 2)
        radian_angle = atan2(dy, dxz)
        return degrees(radian_angle)

video = cv2.VideoCapture(1)

hands = mp.solutions.hands
Hands = hands.Hands(1)
mpDwaw = mp.solutions.drawing_utils
before= point(0,0,0)
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
            landMarks1 = points.landmark[0]#[HandLandmark.WRIST.value]
            landMarks2 = points.landmark[5]#[HandLandmark.INDEX_FINGER_MCP.value]
            landMarks3 = points.landmark[17]#[HandLandmark.PINKY_MCP.value]
            newpoint = point(np.float32((landMarks1.x + landMarks2.x + landMarks3.x))/3,
                            np.float32((landMarks1.y + landMarks2.y + landMarks3.y))/3,
                            np.float32((landMarks1.z + landMarks2.z + landMarks3.z))/3)
            

            cv2.circle(img,(int(newpoint.x * w), int(newpoint.y * h)),4,(0,0,255),cv2.FILLED)

            distancia = newpoint.calculoDistancia(before)
            angulo = newpoint.calculoAngulo(landMarks1)

            tempo_decorrido = time.time() - tempo_inicial

            before = newpoint

            if tempo_decorrido != 0:
                velocidade = distancia/tempo_decorrido
            cv2.putText(img,f"Velocidade: {velocidade:.2f} m/s",(0,50),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),5)
            cv2.putText(img,f"Angulo: {angulo:.2f} graus",(0,100),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),5)

    
    cv2.imshow('Imagem',img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()

#codigo para indicar numeradores para pontos da mao usando for
        #for points in handPoints:

'''
            for id, cord in enumerate(points.landmark):
                cx, cy = int(cord.x * w), int(cord.y * h)
                cv2.putText(img, str(id), (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                #cv2.circle(img,(cx,cy),15,(0,255,0),cv2.FILLED)
                if id == 0:
                    p0 = cord
                elif id == 5:
                    p1 = cord
                elif id == 17:
                    newpoint = point(np.float32((p1.x + cord.x + p0.x)/3),
                                     np.float32((p1.y + cord.y + p0.y)/3),
                                     np.float32((p1.z + cord.z + p0.z)/3))
                    
                    cv2.circle(img,(int(newpoint.x * w), int(newpoint.y * h)),4,(0,0,255),cv2.FILLED)

                    distancia = newpoint.calculoDistancia(before)
                    angulo = newpoint.calculoAngulo(p0)

                    tempo_decorrido = time.time() - tempo_inicial

                    before = newpoint

                    if tempo_decorrido != 0:
                        velocidade = distancia/tempo_decorrido
                    cv2.putText(img,f"Velocidade: {velocidade:.2f} m/s",(0,50),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),5)
                    cv2.putText(img,f"Angulo: {angulo:.2f} graus",(0,100),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),5)
                pontos.append((cx,cy))
#cv2.circle(img,(cx,cy),15,(0,255,0),cv2.FILLED)
'''
