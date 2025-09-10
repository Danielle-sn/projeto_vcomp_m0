# HDR - High Dynamic Range (Alta faixa dinâmica)

### Como funciona na CM3: 

Faz HDR diretamente no hardware do sensor. Staggered HDR ou uma variação
 - **CAPTURA QUASE SIMULTÂNEA**: Configura diferentes tempos de exposição para diferentes linhas de pixels dentro de um único quadro.
  - **LEITURA INTERCALADA**: Lê um conjunto de linhas com difernetes exposições 
   - **DADOS BRUTOS COMBINADOS**: Envia para o processador do Raspberry Pi um fluxo de dados que já contém as informações desses duas exposições 
        -> Esses dados são processados pelo ISP ( Image Signal Processor) que realiza o **Tone Mapping** (mapeamento de tons). Pega essa enorme quantidade de informações de luz e sombra (a alta faixa dinâmica) e a comprime de forma inteligente para que possa ser exibida em uma tela normal (que tem baixa faixa dinâmica)

OBS: 
     - Bom em cenário de alto constraste 
     - Mantém a saturação e a tonalidade corretas das cores
     - Evidência as textura 

     - Ativar o HDR pode limitar a texa de quadros máxima 

---





