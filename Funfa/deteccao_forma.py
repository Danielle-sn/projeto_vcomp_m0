import cv2
import numpy as np 

captura = cv2.VideoCapture(0)

if captura.isOpened(): 
    
    validacao, frame = captura.read() 

    while validacao: 
        validacao, frame = captura.read()

        if not validacao:
            print("Erro: Nao foi possivel ler o frame da câmera.")
            break

        cv2.imshow("Video da Webcam", frame)

        # 1 - DETECÇÃO POR FORMA 

        # conversão para escala de cinza 
        img_cinza = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        #cv2.imshow("Escala de cinza ", img_cinza)

        # desfoque gaussiano 
        img_desf = cv2.GaussianBlur(img_cinza, (7, 7), 0)  
        #cv2.imshow("desfocado ", img_desf)
        '''parâmetros ( img de entrada ,
          tam do kernel - a tupla deve ser de números ímpares e positivos para garantir um pixel central, 
          desvio para sigmaX. Um sigma maior espalha mais o desfoque, enquanto um sigma menor o concentra mais. 0 - sig que ele calcula o valor ideal de sigma com base no tam do kernel'''
        
        # frame_bi = cv2.bilateralFilter(frame_cinza, 9, 75, 75) # uma ótima opçao para borrar sem perder as bordas, mas o processamento é mais lento 
        # cv2.imshow("2 - Filtro Bilateral Aplicado", frame_bi) 
   #-----------------------------------------                                                  
        #thresholding adaptativo - se adapta a brilhos e sombras locais 

# OPERAÇÃO MORFOLÓGICA PRA LIMPAR O RUÍDO 


        frame_TA = cv2.adaptiveThreshold(
            img_desf,          # Imagem de entrada (em escala de cinza)
            255,                    # Valor máximo a ser atribuído (branco)
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, # O método adaptativo a ser usado
            cv2.THRESH_BINARY_INV,  # Tipo de threshold (invertido neste caso)
            11,                     # Tamanho da vizinhança (blockSize)
            2                       # Constante C a ser subtraída da média
        )
        cv2.imshow("thresholding adaptativo", frame_TA)


        contornos, _ = cv2.findContours(frame_TA, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contornos:
            area = cv2.contourArea(cnt)
            if area > 500:  # ajuste conforme o tamanho esperado do quadrado
                approx = cv2.approxPolyDP(cnt, 0.04*cv2.arcLength(cnt, True), True)

                # Se tiver 4 lados → quadrado
                if len(approx) == 4:
                    cv2.drawContours(frame, [approx], -1, (0, 255, 0), 3)

                    # Pega centro do quadrado
                    M = cv2.moments(cnt)
                    if M["m00"] != 0:
                        cx = int(M["m10"]/M["m00"])
                        cy = int(M["m01"]/M["m00"])
                        cv2.circle(frame, (cx, cy), 5, (255, 0, 0), -1)
                        cv2.putText(frame, "Quadrado detectado", (cx-50, cy-10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
                        
        
        key = cv2.waitKey(5) # faz o frame esperar x milissegundos e armazena a tecla 
        if key == 27: #ESC
              break 



captura.release() # finaliza a conexão com a webcam 
cv2.destroyAllWindows() # fechar a janela 