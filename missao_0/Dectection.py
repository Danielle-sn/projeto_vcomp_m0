import cv2
import numpy as np 

def detectar_quadrado(frame):
    # conversão para escala de cinza 
    img_cinza = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # CLAHE - equalização adaptativa do histograma
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img_clahe = clahe.apply(img_cinza)

    # desfoque gaussiano 
    img_desf = cv2.GaussianBlur(img_cinza, (7, 7), 0)  

    # Canny para bordas
    frame_canny = cv2.Canny(img_desf, 50, 100)
    
    # aqui mudamos para RETR_TREE → retorna contornos + hierarquia
    contornos, hierarquia = cv2.findContours(frame_canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # percorre os contornos
    for i, cnt in enumerate(contornos):
        epsilon = 0.02 * cv2.arcLength(cnt, True) 
        approx = cv2.approxPolyDP(cnt, epsilon , True)
        area = cv2.contourArea(cnt)

        if len(approx) == 4 and area > 500: 
            x,y,w,h = cv2.boundingRect(approx)  

            proporcao = float(w) / h if h != 0 else 0
            if 0.95 <= proporcao <= 1.05:
                
                # acessa hierarquia do contorno atual
                # hierarquia[0][i] = [next, prev, first_child, parent]
                next_cont, prev_cont, child, parent = hierarquia[0][i]

                # por exemplo: se não tem pai → quadrado externo
                if parent == -1:
                    cv2.drawContours(frame, [approx], -1, (0,255,0), 3)
                    cv2.putText(frame, f"Quadrado externo (area={int(area)})", 
                                (x,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
                else:
                    # quadrado interno (filho)
                    cv2.drawContours(frame, [approx], -1, (255,0,0), 2)
                    cv2.putText(frame, f"Quadrado interno (area={int(area)})", 
                                (x,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)

                # centro geométrico
                M = cv2.moments(cnt)
                if M["m00"] != 0:
                    cx = int(M["m10"]/M["m00"]) 
                    cy = int(M["m01"]/M["m00"]) 
                    cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)

    return frame, frame_canny, img_clahe


captura = cv2.VideoCapture(1)

if captura.isOpened(): 
    validacao, frame = captura.read() 

    while validacao: 
        validacao, frame = captura.read()

        if not validacao:
            print("Erro: Nao foi possivel ler o frame da câmera.")
            break

        frame_copia = frame.copy()
        frame_copia, frame_canny, img_clahe = detectar_quadrado(frame_copia)

        cv2.imshow("canny", frame_canny)
        cv2.imshow("resultado final", frame_copia)
        cv2.imshow("CLAHE", img_clahe)

        key = cv2.waitKey(5) 
        if key == 27: #ESC
            break 

captura.release() 
cv2.destroyAllWindows() 
