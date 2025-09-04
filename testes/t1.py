#código comentado com o estudo guiado
#o código não funciona :)
import cv2
import numpy as np

captura = cv2.VideoCapture(0)

if captura.isOpened(): 
    
    validacao, frame = captura.read() # output - tupla (validação, matriz do frame da tela - cada lista tem 3 valores respectivamente de RGB por pixel [10, 12, 12]) 

    while validacao: #validação para apenas continuar se não houver erro na leitura a webcam 
        validacao, frame = captura.read()

        if not validacao:
            print("Erro: Nao foi possivel ler o frame da câmera.")
            break

        cv2.imshow("Video da Webcam", frame)


        # ----------- DETECÇÃO POR COR ---------------

        # PRÉ PROCESSAMENTO COM FILTRO BILATERAL
        frame_bi = cv2.bilateralFilter(frame, 9, 75, 75) # parâmetros: 
        cv2.imshow("2 - Filtro Bilateral Aplicado", frame_bi) # Suavizar a imagem para reduzir o ruído da textura e do brilho
        '''
        d: Diâmetro de cada vizinhança de pixel.
        sigmaColor: Valor de  σ no espaço de cores. Quanto maior o valor, as cores mais distantes começarão a se misturar.
        sigmaSpace: Valor de σ no espaço de coordenadas. Quanto maior o seu valor, mais pixels se misturarão, visto que suas cores estão dentro do intervalo sigmaColor. Influência dos pixels pela distância'''

        # BGR PARA HSV 
        frame_hsv = cv2.cvtColor(frame_bi, cv2.COLOR_BGR2HSV) # detecção de cor robusta contra sombras e luz forte.
        cv2.imshow("3 - Imagem HSV", frame_hsv)

        # MÁSCARA PARA LIMITAR O RECONHECIMENTO DA COR #2B0000
        # A cor #2B0000 (R:43, G:0, B:0) em HSV é aproximadamente H:0, S:255, V:43.
        # dentro do limite ficara branco (255) e fora preto (0)
        # aplicação de duas máscaras pois a matiz do vermelho está no  0 e no 180 
        lim_inf1 = np.array([0, 100, 20])
        lim_sup1 = np.array([10, 255, 100]) 

        lim_inf2 = np.array([170, 100, 20])
        lim_sup2 = np.array([180, 255, 100]) 
        
        mascara1 = cv2.inRange(frame_hsv, lim_inf1, lim_sup1)
        mascara2 = cv2.inRange(frame_hsv, lim_inf2, lim_sup2)

        mascara = cv2.bitwise_or(mascara1, mascara2) #une as duas máscara binárias
        cv2.imshow("4 - Mascara de Cor (Bruta)", mascara)

        # REMOÇÃO DE RUÍDOS COM OPERAÇÕES MORFOLÓGICAS
        kernel = np.ones((5,5), np.uint8) # analisa a vizinhança de cda pixel 
        '''
        erosão --> reduz as regiões brancas do primeiro plano 
        dilatação --> expande as regiões brancas do primeiro plano 
        '''
        # ABERTURA : tirar os ruídos brancos do preto. Remove ruídos externos (erosão -> dilatação)
        mascara_limpa = cv2.morphologyEx(mascara, cv2.MORPH_OPEN, kernel)

        # FECHAMENTO: tirar os ruídos pretos do branco. Preenche buracos internos (dilatação -> erosão)
        mascara_limpa = cv2.morphologyEx(mascara_limpa, cv2.MORPH_CLOSE, kernel)
        cv2.imshow("5 - Após as operações morfológicas", mascara_limpa)

        # # ---------- DETECÇÃO DO CONTORNO -------------




        key = cv2.waitKey(5) # faz o frame esperar x milissegundos e armazena a tecla 
        if key == 27: #ESC
            break 

    

captura.release() # finaliza a conexão com a webcam 
cv2.destroyAllWindows() # fechar a janela 