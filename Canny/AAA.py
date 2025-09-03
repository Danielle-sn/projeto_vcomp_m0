import cv2
import numpy as np

# Carrega a imagem que você enviou
# Certifique-se de que o nome do arquivo corresponde ao que você salvou
image = cv2.imread(r'C:\Users\arthu\Downloads\sem_titulo.jpg') 

# Verifica se a imagem foi carregada corretamente
if image is None:
    print("Erro: Não foi possível carregar a imagem. Verifique o caminho/nome do arquivo.")
else:
    # Converte a imagem para escala de cinza
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Passo 1: Suavização com o filtro Gaussiano
    # Experimente com diferentes tamanhos de kernel (ex: (3,3), (5,5), (7,7))
    # Um kernel maior suaviza mais e remove mais ruído/detalhes finos.
    borrada = cv2.GaussianBlur(gray_image, (5, 5), 0)

    # Passo 2: Aplicação do Canny para detectar bordas
    # Experimente com diferentes limiares.
    # Limiares mais baixos capturam mais bordas (inclusive ruído).
    # Limiares mais altos capturam apenas as bordas mais fortes e proeminentes.
    threshold1 = 50
    threshold2 = 150
    bordas = cv2.Canny(borrada, threshold1, threshold2)

    # Exibe a imagem original e as bordas detectadas
    cv2.imshow('Imagem Original', image)
    cv2.imshow('Bordas Detectadas (Canny)', bordas)

    cv2.waitKey(0)
    cv2.destroyAllWindows()