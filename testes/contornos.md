# CONTORNOS 

Contornos podem ser explicados simplesmente como uma curva que une todos os pontos contínuos (ao longo da fronteira), com a mesma cor ou intensidade. Os contornos são uma ferramenta útil para análise de formas e detecção e reconhecimento de objetos.

contours, hierarchy = cv.findContours(image, mode, method)

contornos, hierarquia = cv.findContours (
    thresh, ***o primeiro é a imagem binária de origem***
    cv.RETR_TREE, ***o segundo é o modo de recuperação de contornos que retorna todos os contornos e organiza-os em uma hierarquia***
    cv.CHAIN_APPROX_SIMPLE ***e o terceiro é o método de aproximação de contornos que remove os pontos redundantes e comprime o contorno, economizando memória***
    )
contornos - ***Uma lista de todos os contornos encontrados na imagem. Cada contorno é representado como uma hierarquia ***

hierarquia - ***Uma matriz que descreve a hierarquia dos contornos, fornecendo informações sobre a relação entre eles (pai e filho)***

### Parâmetro MODE

Especifica como os contornos serão recuperados . Ele define a hierarquia dos contornos e pode assumir os seguintes valores

- RETR_EXTERNAL: contornos externos 
- RETR_TREE: contornos externos e internos, fazendo uma árvores com hierarquia desses contornos ( os contornos que contém contornos internos (filhos) são contabilizado duas vezes)
- RETR_LIST: retorna todos os contornos sem hierarquia 
- RETR_CCOMP: recupera todos os contornos e organiza em uma hierarquia de dois níveis. No nível superior estão os contornos externos e no segundo nível os contornos dos buracos dentro desses componentes
