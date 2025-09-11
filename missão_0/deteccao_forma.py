import cv2
import numpy as np 


def detectar_quadrado(frame):
    # conversão para escala de cinza 
    img_cinza = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #CLAHE
    clahe= cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img_clahe = clahe.apply(img_cinza) #Aplica contraste local sem estourar áreas mais claras das quais podem influenciar negativamente na detecção

    # desfoque gaussiano 
    img_desf = cv2.GaussianBlur(img_cinza, (7, 7), 0)  
                                             
    #thresholding adaptativo - se adapta a brilhos e sombras locais 

    # frame_TA = cv2.adaptiveThreshold(
    #     img_desf,          # Imagem de entrada (em escala de cinza)
    #     255,                    # Valor máximo a ser atribuído (branco)
    #     cv2.ADAPTIVE_THRESH_GAUSSIAN_C, # O método adaptativo a ser usado
    #     cv2.THRESH_BINARY_INV,  # Tipo de threshold (invertido neste caso)
    #     11,                     # Tamanho da vizinhança (blockSize)
    #     2                       # Constante C a ser subtraída da média
    # )
    

    frame_canny = cv2.Canny(
            img_desf, # Imagem de entrada (em escala de cinza)
            50,            # Primeiro threshold (threshold1)
            100            # Segundo threshold (threshold2)
        )
    
    contornos, _ = cv2.findContours(frame_canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contornos:
        epsilon = 0.02* cv2.arcLength(cnt, True) #distância máxima do contorno ao contorno aproximado
        approx = cv2.approxPolyDP(cnt, epsilon , True)
        area = cv2.contourArea(cnt)

        # Filtro de quadrado
        if len(approx) == 4 and area > 500: # área em pixels, calcular qual valor seria ideal 
            print(area) # para ver qual seria o número bom p comparar a area
            x,y,w,h = cv2.boundingRect(approx)  # calcula um retângulo reto (não rotacionado) de menor área possível  
            # x - canto superior esquerdo 
            # y -  canto superior esquerdo
            # w - largura em pixels 
            # h - altura em pixels 

            proporcao = float(w) / h if h != 0 else 0
            if 0.95 <= proporcao <= 1.05:
                
                cv2.drawContours(frame, [approx], -1, (0,255,0), 3)
                cv2.putText(frame, "Quadrado", (x,y-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
                # Centro do quadrado
                M = cv2.moments(cnt) # retorna um dicionário com vários momentos
                if M["m00"] != 0: # m00 representa a área do contorno
                    cx = int(M["m10"]/M["m00"]) # m10 soma de todas as coord x 
                    cy = int(M["m01"]/M["m00"]) # m01 soma de todas as coord y 
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

        
        #cv2.imshow("thresholding adaptativo", frame_TA)
        cv2.imshow("canny", frame_canny)
        cv2.imshow("resultado final", frame_copia)
        cv2.imshow("CLAHE", img_clahe)

        
        key = cv2.waitKey(5) # faz o frame esperar x milissegundos e armazena a tecla 
        if key == 27: #ESC
              break 

captura.release() # finaliza a conexão com a webcam 
cv2.destroyAllWindows() # fechar a janela 