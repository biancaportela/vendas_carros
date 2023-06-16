import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns

# Carregar o dataset
@st.cache_data
def get_data():
    url = "dados/df_clean.csv"
    df = pd.read_csv(url)
    return df

# Carrega os dados
df = get_data()



st.title(':mag: Entendendo o dataset')
st.markdown("---")
st.markdown("""
            
            O mercado de carros usados é uma indústria em constante crescimento. Os preços mais acessíveis dos veículos seminovos e usados frente aos modelos zero quilômetro contribuem para a expansão desse segmento, principalmente no Brasil, local em que os preços dos carros novos estão muito elevados.
           
            O objetivo deste projeto foi criar um modelo preditivo que fornecesse aos clientes as ferramentas certas para guiá-las em sua experiência de compra e a tomarem decisões mais bem informadas. Esta página traz gráficos e descrição do dataset utilizado para treinar o modelo.
           
            O surgimento de portais online como o Craiglist tem facilitado boas informações aos clientes e vendedores sobre as tendências e padrões que determinam o valor do carro usado no mercado.
           
            O dataset utilizado consiste de um webscrapping deste portal.A Craigslist é uma rede de comunidades online centralizadas que disponibiliza anúncios dos mais diversos tipos gratuitos aos usuários. Os dados foram raspados durante alguns meses e contém informações sobre carros usados na região dos Estados Unidos. Ele contém  426.880 observações e 26 colunas.
           
            O dataset foi atualizado pela última vez há 2 anos atrás.


            Um overview das primeiras 20 linhas do dataset:

            """
)

st.dataframe(df.head(20))

st.markdown("""
            
           Abaixo você pode visualizar a distribuição das principais categorias do dataset, assim como gráficos de dispersão e correlação.
            
            """
)

st.title('Histogramas')

st.markdown("""
            
            O histograma nos mostra  uma distribuição de frequências, onde a base de cada uma das barras representa uma classe, e a altura a quantidade ou frequência absoluta com que o valor da classe ocorre.
           
            Ele ajuda a visualizar e resumir grandes conjuntos de dados gráficos em variáveis contínuas. Abaixo temos os histogramas de preço e odômetro, após terem sido tratados para eliminação de valores discrepantes.
           
            A distribuição das duas colunas é assimétrica, mostrando que alguns outliers permanecem no dataset.

            """)

######### HISTOGRAMAS


def plot_histogram(data, title, xlabel, ylabel):
    fig = go.Figure()

    fig.add_trace(
        go.Histogram(
            x=data,
            nbinsx=30,
            marker_color='#5e548e',
            opacity=0.75
        )
    )

    fig.update_layout(
        title=title,
        xaxis_title=xlabel,
        yaxis_title=ylabel,
        bargap=0.1
    )

    return fig


# Renderizar o primeiro histograma
fig1 = plot_histogram(df['price'], 'Histograma do Preço', 'Preço', 'Frequência')
st.plotly_chart(fig1)

# Renderizar o segundo histograma
fig2 = plot_histogram(df['odometer'], 'Histograma do Odômetro', 'Odômetro', 'Frequência')
st.plotly_chart(fig2)

st.title('Variáveis categóricas')

st.markdown("""
            Abaixo temos gráficos que mostram como as variáveis categórias se distrubuem. Nos menus, você pode selecionar a coluna de interesse e entender como esses dados se distribuem:
            
            """)

##### TREE MAP


def plot_treemap(data, title):
    labels = data.index
    values = data.values

    colors = ['#231942', '#3a7ca5', '#00a896', '#be95c4', '#81c3d7', '#0a2472', '#2DD881', '#1C0B19', '##A67F8E','#FCF6BD', '#FF99C8', '#E84855','#111D13']

    fig = go.Figure(go.Treemap(
        labels=labels,
        parents=[''] * len(labels),
        values=values,
        marker=dict(
            colors=colors,
            line=dict(width=2),
            colorscale='Viridis'
        ),
        texttemplate='%{label}<br>%{value}',
        textposition='middle center'
    ))

    fig.update_layout(
        title=f'Valores para a Coluna "{coluna_selecionada}" - Treemap',
        autosize=True
    )

    st.plotly_chart(fig)

coluna_selecionada = st.selectbox('Selecione a coluna:', ['condition','cylinders', 'paint_color', 'type'])

if coluna_selecionada:
    contagem_valores = df[coluna_selecionada].value_counts()

    plot_treemap(contagem_valores, coluna_selecionada)

##### GRAFICOS DE BARRAS


coluna_selecionada = st.selectbox('Selecione a coluna:', ['manufacturer', 'state', 'model'])

if coluna_selecionada:
    contagem_valores = df[coluna_selecionada].value_counts()

    top_10 = contagem_valores.sort_values(ascending=False)[:10][::-1]

    fig = go.Figure(go.Bar(
        x=top_10.values,
        y=top_10.index,
        orientation='h',
        marker=dict(color='#5e548e')
    ))

    fig.update_layout(
        title=f'Top 10 Valores para a Coluna "{coluna_selecionada}"',
        xaxis_title='Contagem',
        yaxis_title=coluna_selecionada,
        bargap=0.1
    )
    st.plotly_chart(fig)
    
    
    ## Gráficos verticais
    coluna_selecionada = st.selectbox('Selecione a coluna:', ['fuel', 'title_status', 'transmission', 'drive'])
    
    if coluna_selecionada:
        contagem_valores = df[coluna_selecionada].value_counts()

        valores = contagem_valores.sort_values(ascending=False)
        fig = go.Figure(go.Bar(
        x=valores.index,
        y=valores.values,
        marker=dict(color='#5e548e')
    ))
        fig.update_layout(
        title=f'Valores para a Coluna "{coluna_selecionada}"',
        xaxis_title='Contagem',
        yaxis_title=coluna_selecionada,
        bargap=0.1
    )
        
        st.plotly_chart(fig)


st.title('Gráficos de dispersão')

st.markdown("""
            Gráficos de dispersão nos permitem visualizar como duas variáveis se relacionam. Ao plotar os pontos no gráfico, é possível detectar padrões e tendências nos dados e identificar possíveis correlações entre as variáveis.
           
            Abaixo há dois gráficos de dispersão: um que mostra a relação entre preço e a quantidade de milhas rodadas (odômetro) e outro que indica a relação entre o preço e o ano do carro.

            """)
######## GRAFICOS DE DISPERSÃO
import matplotlib.pyplot as plt
plt.style.use('dark_background')

def scatterplot(x, y, title, xlabel, ylabel):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(x, y, c='#5a189a', alpha=0.5)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    st.pyplot(fig)

scatterplot(df['odometer'], df['price'], 'Relação entre Preço e Odômetro', 'Odômetro', 'Preço')

st.markdown("""
           No gráfico acima, que mostra a relação entre o preço e odômetro, podemos ver claramente uma relação negativa entre as duas variáveis. O gráfico parece indicar que conforme a quantidade de milhas rodadas aumenta, menor o preço de um carro.
           
           No gráfico abaixo, que mostra a relação entre o preço e o ano do carro temos a relação contrária: quanto mais novo é o carro, maior o seu preço:
            """)

scatterplot( df['year'],df['price'], 'Relação entre Preço e Ano', 'Ano', 'Preço')
#####CORRELAÇÃO

st.title('Correlação')

st.markdown("""
           O objetivo da análise de correlação é verificar se existe alguma associação entre as variáveis e, em caso positivo, qual a extensão dessa associação.
           
           O coeficiente de correlação permite calcular a direção e o grau de associação entre as variáveis. A covariância dos desvios padronizados é chamado de coeficiente de correlação de Pearson.Este coeficiente é a medida mais utilizada para a verificação preliminar da relação entre duas variáveis.
           
           Entretanto, o coeficiente de correlação de Pearson é sensível à presença de outliers. Nesse caso, podemos utilizar o Spearman's Rho, que não requer que os dados apresentem distribuição normal. O Spearman's Rho mede o grau da relação monotônica entre duas variáveis é ele que utilizamos nesta análise:
            
            """)

selected_columns = ['price', 'odometer', 'year']
corr_matrix = df[selected_columns].corr(method='spearman')
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.set_theme(style="whitegrid")
sns.set(rc={'figure.figsize': (8, 6)})
fig, ax = plt.subplots()
sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='RdBu', cbar=False)
ax.set_facecolor('black')
ax.tick_params(colors='white')
plt.setp(ax.spines.values(), color='white')
st.pyplot(fig)


st.markdown("""
            - **Correlação entre "price" e "odometer"**: Existe uma correlação negativa moderada entre o preço e a milhagem. Isso significa que, em geral, à medida que a milhagem aumenta, o preço tende a diminuir. No entanto, a relação não é linear, mas sim monotônica.

            - **Correlação entre "price" e "year"**: Existe uma correlação positiva forte entre o preço e o ano do veículo. Isso indica que, em geral, à medida que o ano do veículo aumenta, o preço também tende a aumentar. Novamente, essa relação não é necessariamente linear, mas monotônica.

            - **Correlação entre "odometer" e "year"**: Existe uma correlação negativa forte entre a milhagem e o ano do veículo. Isso significa que, em geral, à medida que o ano do veículo aumenta, a quantidade de milhas percorridas tende a diminuir. Essa relação também é monotônica e não linear.
            
            """)