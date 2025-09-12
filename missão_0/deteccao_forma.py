import cv2
import numpy as np 

# Parâmetros de pré-processamento
CLAHE_CLIPLIMIT = 2.0
CLAHE_GRID_SIZE = (8, 8)
GAUSSIAN_BLUR_KSIZE = (7, 7)

# Parâmetros de filtragem de contornos
CONTOUR_EPSILON = 0.02       # Fator para o approxPolyDP (2%)
MIN_AREA = 500               # Área mínima para ser considerado um quadrado
MIN_ASPECT_RATIO = 0.95      # Proporção mínima 
MAX_ASPECT_RATIO = 1.05      # Proporção máxima

# Parâmetros de desenho
FONT = cv2.FONT_HERSHEY_SIMPLEX
COR_QUADRADO = (0, 255, 0)
COR_CENTRO = (0, 0, 255)


def detectar_quadrado(frame):
    
    frame_cinza = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # conversão para escala de cinza

    clahe= cv2.createCLAHE(CLAHE_CLIPLIMIT, CLAHE_GRID_SIZE)
    frame_clahe = clahe.apply(frame_cinza) #Aplica contraste local sem estourar áreas mais claras das quais podem influenciar negativamente na detecção
   
    frame_suave = cv2.GaussianBlur(frame_clahe, GAUSSIAN_BLUR_KSIZE, 0)   # desfoque gaussiano 
                                             
    bordas_canny = cv2.Canny( frame_suave, 50, 100)
    

    contornos, hierarquia = cv2.findContours(bordas_canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) # RETR_TREE → retorna contornos + hierarquia

    quadrados_verificados = []

    for i, cnt in enumerate(contornos):

        area = cv2.contourArea(cnt)

        perimetro = cv2.arcLength(cnt, True)
        epsilon = CONTOUR_EPSILON * perimetro #distância máxima do contorno ao contorno aproximado
        approx = cv2.approxPolyDP(cnt, epsilon , True)

        # Filtro de quadrado
        if len(approx) == 4 and area > MIN_AREA: # área em pixels, calcular qual valor seria ideal 
            print(area) 
            x,y,w,h = cv2.boundingRect(approx)  # calcula um retângulo reto (não rotacionado) de menor área possível  
            espectro_ratio = float(w) / h if h != 0 else 0

            if MIN_ASPECT_RATIO <= espectro_ratio <= MAX_ASPECT_RATIO:
                # acessa hierarquia do contorno atual
                # hierarquia~[0][i] = [next, prev, first_child, parent]
                _, _, _, parent = hierarquia[0][i]

                # ex: Se não tem pai -> quadrado externo
                if parent == -1:
                    #continue
                    cv2.drawContours(frame, [approx], -1, (0,255,0), 3)
                    cv2.putText(frame, "Quadrado externo", (x,y-10), FONT, 0.7, (0,255,0), 2)

                else: 
                    # quadrado interno (filho)
                    quadrados_verificados.append(approx)       
                   

    return quadrados_verificados, bordas_canny, frame_clahe

def desenhar(frame, quadrados):
    for quadrado in quadrados:
        cv2.drawContours(frame, [quadrado], -1, (255,0,0), 3 )

        x, y, _, _ = cv2.boundingRect(quadrado)
        cv2.putText(frame, "Quadrado interno", (x, y - 10), FONT, 0.7, COR_QUADRADO, 2)

         # Cálculo e desenho do centro do quadrado
 
        M = cv2.moments(quadrado) # retorna um dicionário com vários momentos
        if M["m00"] != 0: # m00 representa a área do contorno
                    cx = int(M["m10"]/M["m00"]) # m10 soma de todas as coord x 
                    cy = int(M["m01"]/M["m00"]) # m01 soma de todas as coord y 
                    cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)


def main():

    captura = cv2.VideoCapture(1)
        
    validacao, frame = captura.read() 

    while validacao: 
        validacao, frame = captura.read()

        if not validacao:
            print("Erro: Nao foi possivel ler o frame da câmera.")
            break

        frame_copia = frame.copy()

        quadrados_verificados, bordas_canny, img_clahe = detectar_quadrado(frame_copia)

        desenhar(frame_copia, quadrados_verificados)
        
        #cv2.imshow("thresholding adaptativo", frame_TA)
        cv2.imshow("canny", bordas_canny)
        cv2.imshow("resultado final", frame_copia)
        cv2.imshow("CLAHE", img_clahe)

        
        key = cv2.waitKey(5) # faz o frame esperar x milissegundos e armazena a tecla 
        if key == 27: #ESC
            break 

    captura.release() # finaliza a conexão com a webcam 
    cv2.destroyAllWindows() # fechar a janela 

if __name__ == "__main__":
    main()