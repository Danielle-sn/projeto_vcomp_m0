import cv2
import numpy as np

def detectar_quadrado(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Faixa 1 - vermelho claro
    lower1 = np.array([0, 100, 100])
    upper1 = np.array([10, 255, 255])

    # Faixa 2 - vermelho escuro
    lower2 = np.array([170, 100, 100])
    upper2 = np.array([180, 255, 255])

    mask1 = cv2.inRange(hsv, lower1, upper1)
    mask2 = cv2.inRange(hsv, lower2, upper2)
    mask = cv2.bitwise_or(mask1, mask2)

    # Limpeza
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5,5),np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, np.ones((5,5),np.uint8))

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        approx = cv2.approxPolyDP(cnt, 0.02*cv2.arcLength(cnt, True), True)
        if len(approx) == 4 and cv2.contourArea(cnt) > 500:
            cv2.drawContours(frame, [approx], -1, (0,255,0), 3)
            x,y,w,h = cv2.boundingRect(approx)
            cv2.putText(frame, "Quadrado detectado", (x,y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

    return frame, mask


cap = cv2.VideoCapture(1)  # 0 = webcam, 1 = celular ou outra cam e assim por diante

while True:
    ret, frame = cap.read()
    if not ret or frame is None:
        continue

    resultado, mask = detectar_quadrado(frame)

    cv2.imshow("Deteccao", resultado)
    cv2.imshow("Mascara", mask)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()
