
import cv2
import numpy as np
fotoManeira = r'C:\Users\arthu\Downloads\sem_titulo.jpg'
camera = cv2.VideoCapture

imagem = cv2.imshow(camera)
cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)

borrao = cv2.GaussianBlur(cinza, (5,5),0)

threshold1 = 50
threshold2 = 170
edged = cv2.Canny(borrao, threshold1,threshold2)

cv2.imshow("Imagem brutal", borrao)
cv2.imshow("Imagem com as bordas", edged)
cv2.waitKey(0)    
camera.release()
cv2.destroyAllWindows()