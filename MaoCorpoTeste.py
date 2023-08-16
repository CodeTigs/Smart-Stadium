# Importação de bibliotecas
import cv2 # Biblioteca para captura e processamento de imagem
import mediapipe as mp  # Biblioteca para detecção de poses e mãos
from math import sqrt, atan2, degrees  # Funções matemáticas

# Definição de uma classe para representar pontos em 3D
class point:
    def __init__(self, x, y, z):
        self.x, self.y, self.z = x, y, z

    # Método para calcular a distância entre dois pontos em 3D
    def calculoDistancia(self, point):
        return sqrt((self.x - point.x) ** 2 + (self.y - point.y) ** 2 + (self.z - point.z) ** 2)

    # Método para calcular o ângulo entre dois pontos no plano XZ em relacao ao plano Y
    def calculoAngulo(self, point):
        dy = abs(point.y - self.y)
        dxz = sqrt((self.x - point.x) ** 2 + (self.z - point.z) ** 2)
        radian_angle = atan2(dy, dxz)
        return degrees(radian_angle)

# Configuração do mediapipe para detecção de poses e mãos
mp_pose = mp.solutions.pose  # Módulo para detecção deo corpo
mp_hand = mp.solutions.hands  # Módulo para detecção de mãos
mp_drawing = mp.solutions.drawing_utils  # Utilidades para desenhar pontos e conexões nas imagens

# Configuração dos modelos de detecção de corpo e mãos
pose = mp_pose.Pose(
    min_detection_confidence=0.5,  # Limiar de confiança mínimo para detecção do corpo
    min_tracking_confidence=0.5)   # Limiar de confiança mínimo para o rastreamento do corpo
hand = mp_hand.Hands()  # Objeto para detecção de mãos

# Inicialização da câmera de vídeo
video = cv2.VideoCapture(0)

# Loop principal para processar cada quadro do vídeo
while True:
    success, img = video.read()  # Captura de um quadro da câmera
    img = cv2.flip(img, 1)  # Espelhar horizontalmente a imagem capturada
    frameRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Converter de BGR para RGB
    h, w, _ = img.shape  # Obter as dimensões da imagem

    try:
        # Processamento de pontos no corpo
        results = pose.process(frameRGB)  # Processar o quadro para detectar o corpo
        posePoints = results.pose_landmarks  # Pontos chave do corpo detectado
        if posePoints is not None:
            # Desenhar conexões entre os pontos chave do corpo na imagem
            mp_drawing.draw_landmarks(img, posePoints, mp_pose.POSE_CONNECTIONS)
    except:
        pass

    try:
        # Processamento de pontos nas mãos
        results2 = hand.process(frameRGB)  # Processar o quadro para detectar mãos
        handPoints = results2.multi_hand_landmarks  # Pontos chave nas mãos detectadas
        if handPoints is not None:
            # Desenhar conexões entre os pontos chave nas mãos na imagem
            for points in handPoints:
                mp_drawing.draw_landmarks(img, points, mp_hand.HAND_CONNECTIONS)
    except:
        pass

    cv2.imshow('Image', img)  # Mostrar a imagem resultante com as marcas e conexões
    if cv2.waitKey(1) & 0xFF == ord('q'):  # Esperar a tecla 'q' para sair do loop
        break

video.release()  # Liberar a câmera
cv2.destroyAllWindows()  # Fechar todas as janelas abertas criadas pelo cod
