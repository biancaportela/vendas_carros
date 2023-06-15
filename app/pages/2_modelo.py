import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image


# Carregar o dataset
@st.cache_data
def get_data():
    url = "dados/df_clean.csv"
    df = pd.read_csv(url)
    return df

# Carrega os dados
df = get_data()



st.title(':deciduous_tree: Sobre a Random Florest')
st.markdown("---")

st.title('Métricas')
st.markdown("""
            Este dataset foi treinado utilizando vários modelos, sendo eles a Regressão Linear, a Regressão de Huber, a regressão RANSAC, Random Forest e XGBoost. O modelo foi treinado utilizando cross validation com 5 k-folds. Desses modelos a Random Forest obteve o melhor desempenho nas métricas avaliadas. 
            
            **Random Forest**: é um algoritmo de aprendizado de máquina que combina várias árvores de decisão independentes e faz previsões tomando uma média ou votação das previsões individuais das árvores, resultando em um modelo mais robusto e com menor tendência ao overfitting.
            
            As métricas observadas foram:
            
            - **R-quadrado**: O R-quadrado é uma métrica que mede a proporção da variância total dos dados explicada pelo modelo de regressão, indicando o quão bem as variáveis independentes explicam a variabilidade da variável dependente.
            - **MAE**: O MAE é uma métrica que calcula a média dos valores absolutos dos erros entre as previsões do modelo e os valores reais, fornecendo uma medida direta do tamanho médio dos erros de previsão.
            - **RMSE**(Raiz do Erro Quadrático Médio): O RMSE é uma métrica usada para avaliar a precisão de modelos de regressão, que calcula a raiz quadrada da média dos quadrados das diferenças entre os valores previstos e os valores reais.
            - **MAD** (Erro Absoluto Medio): é uma métrica que calcula a media dos valores absolutos dos erros entre as previsões do modelo e os valores reais, proporcionando uma medida robusta e menos sensível a outliers do tamanho médio dos erros de previsão.
            
            Os resultados para os dados de treino em todos os modelos estão no dataframe abaixo, e podemos claramente ver o desempenho superior do modelo de Random Forest:
            
            """
)


dados = [
    {"Modelo": "LinearRegression", "R2": 0.6179628388509528, "MAE": 0.040004748655643234, "RMSE": 0.059722130570298584, "MAD": 0.040004748655643234},
    {"Modelo": "Ridge", "R2": 0.6179628687511066, "MAE": 0.0400047315419372, "RMSE": 0.05972212832872299, "MAD": 0.0400047315419372},
    {"Modelo": "HuberRegressor", "R2": 0.6050541875235333, "MAE": 0.03900465222222114, "RMSE": 0.06072340070641687, "MAD": 0.03900465222222114},
    {"Modelo": "RandomForestRegressor", "R2": 0.8634362967153096, "MAE": 0.016657201209182453, "RMSE": 0.035704024575044166, "MAD": 0.016657201209182453},
    {"Modelo": "XGBRegressor", "R2": 0.8009899991152809, "MAE": 0.026083716193812702, "RMSE": 0.04310273654449514, "MAD": 0.026083716193812702}
]

avaliacao = pd.DataFrame(dados)
pd.options.display.float_format = '{:.3f}'.format

st.dataframe(avaliacao )


st.markdown("""
            Com o modelo final já treinado e com as devidas métricas podemos utilizá-lo para os dados de teste e assim ver o quão bem ele generaliza para dados novos. Na tabela abaixo, temos resultados similares aos obtidos no treino, mostrando que o modelo se generaliza bem.


            """)

data = {
    '': ['Base de Teste'],
    'R2': [0.860],
    'MAE': [0.017],
    'RMSE': [0.036],
    'MAD': [0.017]
}
df = pd.DataFrame(data)

# Exibir o DataFrame no Streamlit
st.table(df)

st.title('Análise de resíduos')


st.markdown("""
            Pelo desempenho superior do modelo de Random Forest, eu também analisei seus resíduos. A análise de resíduos é um tópico clássico relacionado à modelagem estatística e é frequentemente utilizada para avaliar a adequação de um modelo. Dessa maneira, os resíduos são calculados utilizando-se os dados de treinamento e usados para avaliar se as previssões do modelo se ajustam aos valores observados da variável dependente. Os resíduos também podem nos indicar se o modelo tem erros heterocedásticos ou se são afetados por outliers.

            Para a maioria dos modelos, os resíduos devem apresentar um comportamento aleatório com certas propriedades (como, por exemplo, estar concentrados em torno de 0). Se encontrarmos quaisquer desvios sistemáticos do comportamento esperado, eles podem indicar um problema com o modelo (por exemplo, uma variável explicativa omitida ou uma forma funcional incorreta de uma variável incluída no modelo).

            Vemos isso no gráfico abaixo:
            
            """
)

resid_1 = Image.open("imagens/arvore_resid.png")
st.image(resid_1)


st.markdown("""
            Para um modelo "bom", os resíduos devem se desviar aleatoriamente de zero. Assim, sua distribuição deve ser simétrica em torno de zero, o que implica que seu valor médio (ou mediano) deve ser zero. Além disso, os resíduos devem ser próximos de zero em si mesmos, ou seja, devem apresentar baixa variabilidade. O gráfico acima mostra a diferença entre os resíduos (valores observados da variável target - o valor predito) no eixo Y e a variável dependente no eixo X.  A dispersão no painel superior reflete  o aumento da variabilidade dos resíduos para valores ajustados crescentes. Isso indica uma violação da suposição de homocedasticidade, ou seja, da constância da variância.


            Mas quando vemos o histograma, percebemos que os resíduos se concentram em torno de 0:

"""
)

resid_2 = Image.open("imagens/hist_resid.png")
st.image(resid_2)

st.markdown(
            """
            Por fim, fiz um gráfico de erro de previsão que mostra os valores reais do conjunto de dados em relação aos valores previstos gerados pelo nosso modelo. Isso nos permite ver quanto de variação há no modelo.


            Ao exibir esses ajustes de linha no gráfico de erro de previsão, o objetivo é comparar visualmente como as previsões do modelo se comparam aos valores reais. O ajuste de linha "identity" ajuda a identificar se o modelo está sub ou superestimando as previsões de maneira sistemática. O ajuste de linha "best fit" indica a tendência geral do modelo e se ele está sub ou superestimando as previsões de maneira consistente.


            O modelo se ajusta bem para boa parte da amostra, mas perde poder de previsibilidade conforme os valores de y se tornam maiores, subestimando consistentemente os valores reais.

            """
)

resid_3 = Image.open("imagens/pred_error.png")
st.image(resid_3)

st.title('Interpretação da árvore')

st.markdown("""
            Nessa seção eu me debruço em quais as variáveis o Random Florest considerou mais importate para fazer suas previsões. Saber quais variáveis são importantes e quais não são podem ajudar a otimizar o modelo e diminuir o tempo de convergência. 

            Primeiramente, eu plotei uma das árvores de decisão feitas pelo algoritmo de Random Florest para os primeiros 3 níveis, de modo que possamos ter uma noção de como elas se divide. Nela podemos ver que a árvore é decidida por `year`, `drive_fwd`, `odômetro` e `cilindros` nos níveis iniciais.
            
            """)


tree = Image.open("imagens/tree.png")
st.image(tree)

st.markdown("""
            Na visualização dos features importance, optei pelo método da permutação. Este método  embaralha aleatoriamente cada característica e calcula a mudança no desempenho do modelo. As características que têm maior impacto no desempenho são as mais importantes. Esse método apresenta performance melhor ao lidar com dados com alta cardinalidade e com muitas variáveis categóricas.

            
            """)

features = Image.open("imagens/feature importance.png")
st.image(features)

st.markdown("""
            Continuemos com a interpretação dos modelos de regressão. O intercepto dos coeficientes das regressões explicam seu valor esperado e como as features impactam a previsão. Um coeficiente positivo indica que se o valor das features aumenta, o valor predito também aumenta. Nós já checamos a feature importance, mas outro método que podemos utilizar é a interpretação das árvores utilizando o pacote `treeinterpreter`. Este pacote calcula o viés e a contribuição de cada feature no modelo. O viés é a média de todo o conjunto de testes.


            Cada contribuição lista como ela contribui para cada um dos rótulos (o viés mais a contribuição devem somar a previsão). Ao fim obtive o seguinte gráfico:
            
            """)

contribuicao_media = Image.open("imagens/contribuicao_media.png")
st.image(contribuicao_media)


st.markdown("""
            Através desse gráfico, vemos claramente que a variável `odometer` e `drive_fwd` tem as maiores contribuições em fazer com que a árvore preveja que os preços serão mais altos. Já `year` e `fuel_gas` contribuem para que os preços sejam mais baixos.
            
            Se você tem interesse em ver o código mais detalhes, convido-o a checar o [repositório](https://github.com/biancaportela/vendas_carros) do projeto.  Muito obrigada pela leitura!
            
            
            """)