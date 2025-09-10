# CONTORNOS 

Contornos podem ser explicados simplesmente como uma curva que une todos os pontos contínuos (ao longo da fronteira), com a mesma cor ou intensidade. Os contornos são uma ferramenta útil para análise de formas e detecção e reconhecimento de objetos.

contornos, hierarquia = cv.findContours (
    thresh, ***o primeiro é a imagem de origem***
    cv.RETR_TREE, ***o segundo é o modo de recuperação de contornos***
    cv.CHAIN_APPROX_SIMPLE ***e o terceiro é o método de aproximação de contornos***
    )


