import cv2
import numpy as np

# Abre a câmera
camera = cv2.VideoCapture(1)

# CLAHE para lidar com iluminação variável
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))

while True:
    ret, frame = camera.read()
    if not ret:
        break

    # Pré-processamento: CLAHE no espaço LAB
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l_clahe = clahe.apply(l)
    lab_clahe = cv2.merge((l_clahe, a, b))
    frame_eq = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)

    # Converte para HSV
    hsv = cv2.cvtColor(frame_eq, cv2.COLOR_BGR2HSV)

    # --- Faixas de vermelho ---
    # Vermelho escuro da missão (#2B0000) e similares
    low_dark_red = np.array([0, 100, 20])
    upp_dark_red = np.array([10, 255, 120])

    low_dark_red2 = np.array([170, 100, 20])
    upp_dark_red2 = np.array([180, 255, 120])

    # Tons mais claros (condições de luz forte/reflexos)
    low_light_red = np.array([0, 50, 120])
    upp_light_red = np.array([10, 200, 255])

    # Máscaras
    mask1 = cv2.inRange(hsv, low_dark_red, upp_dark_red)
    mask2 = cv2.inRange(hsv, low_dark_red2, upp_dark_red2)
    mask3 = cv2.inRange(hsv, low_light_red, upp_light_red)

    # Combina todas
    mask = cv2.bitwise_or(mask1, mask2)
    mask = cv2.bitwise_or(mask, mask3)

    # Remove ruído
    kernel = np.ones((5, 5), np.uint8)
    mask_clean = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # Resultado segmentado
    result = cv2.bitwise_and(frame_eq, frame_eq, mask=mask_clean)

    # --- Contornos ---
    contornos, _ = cv2.findContours(mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contornos:
        area = cv2.contourArea(cnt)
        if area > 500:  # ajuste para escala 8x8m
            approx = cv2.approxPolyDP(cnt, 0.04*cv2.arcLength(cnt, True), True)

            if len(approx) == 4:  # quadrado
                cv2.drawContours(frame_eq, [approx], -1, (0, 255, 0), 3)

                # Centro
                M = cv2.moments(cnt)
                if M["m00"] != 0:
                    cx = int(M["m10"]/M["m00"])
                    cy = int(M["m01"]/M["m00"])
                    cv2.circle(frame_eq, (cx, cy), 5, (255, 0, 0), -1)
                    cv2.putText(frame_eq, "Quadrado detectado", (cx-50, cy-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

    # Mostra imagens
    cv2.imshow("Camera", frame_eq)
    cv2.imshow("Mascara", mask_clean)
    cv2.imshow("Resultado", result)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()
