import cv2
import numpy as np 
from config import PRE_PROCESSAMENTO, FILTRAGEM_CONTORNOS, DESENHO, ANGULACAO, FILTRO_KALMAN

def incializar_kalman():
    kf = cv2.KalmanFilter(4, 2)

    kf.transitionMatrix = np.array([[1, 0, 1, 0],
                                    [0, 1, 0, 1],
                                    [0, 0, 1, 0],
                                    [0, 0, 0, 1]], np.float32)
    
    kf.measurementMatrix = np.array([[1, 0, 0, 0],
                                     [0, 1, 0, 0]], np.float32)
    
    kf.processNoiseCov = np.eye(4, dtype=np.float32) * FILTRO_KALMAN["uncertainty_magnitude"]

    kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * FILTRO_KALMAN["noise_magnitude"]
    

def detectar_quadrado(frame):
    
    frame_cinza = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # conversão para escala de cinza
    clahe= cv2.createCLAHE(PRE_PROCESSAMENTO["clahe_cliplimit"], PRE_PROCESSAMENTO["clahe_grid_size"])
    frame_clahe = clahe.apply(frame_cinza) #Aplica contraste local sem estourar áreas mais claras das quais podem influenciar negativamente na detecção
    frame_suave = cv2.GaussianBlur(frame_clahe, PRE_PROCESSAMENTO["gaussian_blur_ksize"], 0)   # desfoque gaussiano                                       
    bordas_canny = cv2.Canny( frame_suave, 50, 100)
    

    contornos, hierarquia = cv2.findContours(bordas_canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) # RETR_TREE → retorna contornos + hierarquia

    espectro_ratio = 0
    quadrados_internos = []

    if hierarquia is not None:
        hierarquia = hierarquia[0]
    
        def angulo_cos(p1, p2, p3):
            v1 = p1 - p2
            v2 = p3 - p2
            cosang = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            return np.degrees(np.arccos(cosang))
        
        
        
        for i, cnt in enumerate(contornos):

            area = cv2.contourArea(cnt)
            perimetro = cv2.arcLength(cnt, True)
            epsilon =  FILTRAGEM_CONTORNOS["contour_epsilon"] * perimetro #distância máxima do contorno ao contorno aproximado
            approx = cv2.approxPolyDP(cnt, epsilon , True)

            # Filtro de quadrado
            if len(approx) == 4 and area > FILTRAGEM_CONTORNOS["min_area"]: # área em pixels, calcular qual valor seria ideal 
                print(area) 
                angulos = []
                for j in range(4):
                    p1 = approx[j][0]
                    p2 = approx[(j+1)%4][0]
                    p3 = approx[(j+2)%4][0]
                    angulos.append(angulo_cos(p1, p2, p3))


                if all ( ANGULACAO["lower_limit"] <= ang <= ANGULACAO["upper_limit"] for ang in angulos): #Tolera 10°
                    rect = cv2.minAreaRect(approx)
                    (w, h) = rect[1]

                    if h != 0:
                        espectro_ratio = float(w) / h if w > h else float(h) / w

                        if FILTRAGEM_CONTORNOS["min_aspect_ratio"] <= espectro_ratio <= FILTRAGEM_CONTORNOS["max_aspect_ratio"]:
                            # acessa hierarquia do contorno atual
                            # hierarquia~[0][i] = [next, prev, first_child, parent]
                            parent = hierarquia[i][3]
                            child = hierarquia[i][2]

                            # ex: Se não tem pai -> quadrado externo
                            if parent != -1 and child == -1:
                                quadrados_internos.append((area, approx))           

    return quadrados_internos, bordas_canny, frame_clahe

def desenhar(frame, quadrados):
    for area,quadrado in quadrados:
        cv2.drawContours(frame, [quadrado], -1, (255,0,0), 3 )

        x, y, _, _ = cv2.boundingRect(quadrado)
        cv2.putText(frame, F"Quadrado interno - area:{int(area)}", (x, y - 10), DESENHO["font"], 0.7, DESENHO["cor_quadrado"], 2)

         # Cálculo e desenho do centro do quadrado
 
        M = cv2.moments(quadrado) # retorna um dicionário com vários momentos
        if M["m00"] != 0: # m00 representa a área do contorno
                    cx = int(M["m10"]/M["m00"]) # m10 soma de todas as coord x 
                    cy = int(M["m01"]/M["m00"]) # m01 soma de todas as coord y 
                    cv2.circle(frame, (cx, cy), 5, DESENHO["cor_centro"], -1)


