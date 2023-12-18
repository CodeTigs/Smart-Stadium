import cv2
import mediapipe as mp
from mediapipe.python.solutions.pose import PoseLandmark
from mediapipe.python.solutions.drawing_utils import DrawingSpec
import time
import numpy as np
from math import sqrt,atan2, degrees

class point:
    #construtor com possibidade de entrada com 1 parametro(ponto com posicoes x,y,z) ou
    #com 3 valores de cordenadas(x,y,z) 
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

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

before= point(0,0,0) #iniciando aptura na posicao 0
velocidade = 0.0 #iniciando valor de velocidade como 0
velocidadeMedia = []#lista usada para calcular a velocidade media da mao

#variveis usadas para selecionar os pontos que nao aparecerao na tela
custom_style = mp_drawing_styles.get_default_pose_landmarks_style()
custom_connections = list(mp_pose.POSE_CONNECTIONS)



#pontos do corpo que nao serao capturados
excluded_landmarks = [
    PoseLandmark.LEFT_EYE, 
    PoseLandmark.RIGHT_EYE, 
    PoseLandmark.LEFT_EYE_INNER, 
    PoseLandmark.RIGHT_EYE_INNER, 
    PoseLandmark.LEFT_EAR,
    PoseLandmark.RIGHT_EAR,
    PoseLandmark.LEFT_EYE_OUTER,
    PoseLandmark.RIGHT_EYE_OUTER,
    PoseLandmark.NOSE,
    PoseLandmark.MOUTH_LEFT,
    PoseLandmark.MOUTH_RIGHT,
    PoseLandmark.RIGHT_PINKY,
    PoseLandmark.RIGHT_INDEX,
    PoseLandmark.RIGHT_THUMB,
    PoseLandmark.LEFT_PINKY,
    PoseLandmark.LEFT_INDEX,
    PoseLandmark.LEFT_THUMB
    ]

for landmark in excluded_landmarks:    
    custom_style[landmark] = DrawingSpec(color=(255,255,0), thickness=None) #mudando a forma como os pontos de referência excluídos são desenhados  
    custom_connections = [connection_tuple for connection_tuple in custom_connections 
                            if landmark.value not in connection_tuple] #removendo todas as conexões que contêm esses pontos de referência
    
pose = mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

video = cv2.VideoCapture(0)

while True:
    #captura de imagem
    success, img = video.read()
    img = cv2.flip(img, 1)
    frameRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w, _ = img.shape # pegando proporcoes da tela
    tempo_inicial = time.time()#tempo inicial para calculo de velocidade
    try:
        #gerando imagem com pontos selecionados
        results = pose.process(frameRGB)
        posePoints = results.pose_landmarks

        mp_drawing.draw_landmarks(
            img,
            results.pose_landmarks,
            connections = custom_connections, #  passing the modified connections list
            landmark_drawing_spec=custom_style) # and drawing styledc´

        newpoint = point(posePoints.landmark[30])
        distancia = newpoint.calculoDistancia(before)

        tempo_decorrido = time.time() - tempo_inicial #tempo final para calculo de velocidade

        #condicoes para calculo de velocidade
        if before is not None:
            distancia = newpoint.calculoDistancia(before)
            if tempo_decorrido > 0:
                velocidade = distancia / tempo_decorrido
            else:
                velocidade = 0.0
        else:
            velocidade = 0.0

        cv2.putText(img,f"Velocidade: {velocidade:.2f} m/s",(0,50),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),5)
        before = newpoint #salvando ponto central da mao para calculo de distanca  
        velocidadeMedia.append(velocidade)


    except:
        continue

    cv2.imshow('Image', img) #gerando aba que mostra a captura da camera 
    if cv2.waitKey(1) & 0xFF == ord('q'):#condicao para parar a captura
        break

#encerrando a captura 
video.release()
cv2.destroyAllWindows()
velocidadeMedia = np.mean(velocidadeMedia)
print(f"Velocidade média: {velocidadeMedia:.2f} m/s")

        #colocar dentro do try{
        #pegando pontos para futuramente calcular angulo e velocidade
        #landMarks[PoseLandmark.RIGHT_SHOULDER.value].x
        #landMarks = posePoints.landmark 

        #captura convencional de pontos
        #mp_drawing.draw_landmarks(img, posePoints, mp_pose.POSE_CONNECTIONS)
        #}
        #numerando pontos 
        #for id, cord in enumerate(posePoints.landmark):
            #cx, cy = int(cord.x * w), int(cord.y * h)
            #cv2.putText(img, str(id), (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)