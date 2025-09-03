import cv2
import numpy as np

# Abre a câmera (0 = câmera padrão)
camera = cv2.VideoCapture(0)


# Definindo thresholds
threshold1 = 50
threshold2 = 170

while True:
    # Captura frame da câmera
    ret, frame = camera.read()
    if not ret:
        break

    # Converte para escala de cinza
    cinza = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Aplica blur
    borrao = cv2.GaussianBlur(cinza, (5,5), 0)

    # Detecção de bordas
    edged = cv2.Canny(borrao, threshold1, threshold2)

    # Mostra as imagens
    cv2.imshow("Camera", frame)
    cv2.imshow("Borrada", borrao)
    cv2.imshow("Bordas", edged)

    # Sai do loop quando aperta a tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libera a câmera e fecha janelas
camera.release()
cv2.destroyAllWindows()
