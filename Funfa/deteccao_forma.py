import cv2
import numpy as np 


def detectar_quadrado(frame):
    # conversão para escala de cinza 
    img_cinza = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    # desfoque gaussiano 
    img_desf = cv2.GaussianBlur(img_cinza, (7, 7), 0)  
                                             
    #thresholding adaptativo - se adapta a brilhos e sombras locais 

    frame_TA = cv2.adaptiveThreshold(
        img_desf,          # Imagem de entrada (em escala de cinza)
        255,                    # Valor máximo a ser atribuído (branco)
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, # O método adaptativo a ser usado
        cv2.THRESH_BINARY_INV,  # Tipo de threshold (invertido neste caso)
        11,                     # Tamanho da vizinhança (blockSize)
        2                       # Constante C a ser subtraída da média
    )
    

    frame_canny = cv2.Canny(
            img_desf, # Imagem de entrada (em escala de cinza)
            50,            # Primeiro threshold (threshold1)
            100            # Segundo threshold (threshold2)
        )
    
    contornos, _ = cv2.findContours(frame_canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contornos:
        approx = cv2.approxPolyDP(cnt, 0.02*cv2.arcLength(cnt, True), True)
        area = cv2.contourArea(cnt)

        # Filtro de quadrado
        if len(approx) == 4 and area > 500:
            x,y,w,h = cv2.boundingRect(approx)
            cv2.drawContours(frame, [approx], -1, (0,255,0), 3)
            cv2.putText(frame, "Quadrado", (x,y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
    return frame,frame_TA, frame_canny


captura = cv2.VideoCapture(1)

if captura.isOpened(): 
    
    validacao, frame = captura.read() 

    while validacao: 
        validacao, frame = captura.read()

        if not validacao:
            print("Erro: Nao foi possivel ler o frame da câmera.")
            break
        frame_copia = frame.copy()

        frame_copia,frame_TA, frame_canny = detectar_quadrado(frame_copia)

        
        cv2.imshow("thresholding adaptativo", frame_TA)
        cv2.imshow("canny", frame_canny)
        cv2.imshow("resultado final", frame_copia)
    

        
        
        key = cv2.waitKey(5) # faz o frame esperar x milissegundos e armazena a tecla 
        if key == 27: #ESC
              break 

captura.release() # finaliza a conexão com a webcam 
cv2.destroyAllWindows() # fechar a janela 