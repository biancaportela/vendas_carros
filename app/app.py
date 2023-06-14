import streamlit as st
import pickle
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler

############  CARREGANDO MODELO E DATASET ##########

# Carregar o modelo treinado
with open('sales_rfr.pkl', 'rb') as file:
    model = pickle.load(file)


# Carregar o dataset
@st.cache_data
def get_data():
    df = pd.read_csv("C:/Users/SAMSUNG/OneDrive/Documentos/GitHub/vendas_carros/dados/df_clean.csv")
    
    return df

# Função para verificar se um valor é numérico 
def is_numeric(value):
    try:
        float(value)
        return True
    except ValueError:
        return False

df = get_data()
##########  LIMPEZA DOS DADOS ###########

df = df.drop(columns=['region'])
df = df.drop(columns=['model'])
#categorizando estados por macro região
estados_nordeste = ['me', 'nh','vt','ma', 'ri', 'ct', 'ny', 'pa', 'nj']
nordeste = 'nordeste'
estados_centro_oeste = ['wi', 'mi', 'il', 'in', 'oh', 'mo', 'sd', 'nd', 'ne', 'ks', 'mn', 'ia']
centro_oeste = 'centro oeste'
estados_sul = ['de', 'md', 'va', 'wv', 'nc', 'sc', 'ga', 'fl', 'ky', 'tn', 'ms', 'al', 'ok', 'tx', 'ar', 'la', 'dc']
sul  = 'sul'
estados_oeste = ['id', 'mt', 'wy', 'nv', 'ut', 'co', 'az', 'nm', 'ca', 'or', 'wa', 'ak', 'hi']
oeste = 'oeste'
df['state'] = df['state'].replace(estados_nordeste, nordeste)
df['state'] = df['state'].replace(estados_centro_oeste, centro_oeste)
df['state'] = df['state'].replace(estados_sul, sul)
df['state'] = df['state'].replace(estados_oeste, oeste )
#categorizando manufacturer
carros = ['volvo', 'mitsubishi', 'mini', 'pontiac', 'jaguar', 'rover', 'porsche', 'saturn', 'mercury', 'alfa-romeo', 'tesla', 'fiat', 'harley-davidson', 'datsun', 'ferrari', 'aston-martin', 'land rover']
others = 'others'
df['manufacturer'] = df['manufacturer'].replace(carros, others)

X = df.drop(columns=['price'])
y = df['price']

####################### PRE PROCESSAMENTO #########

# Separando as variáveis que receberão one hot encoding e as que receberão normalização
onehot_cols = ['condition', 'cylinders', 'fuel', 'title_status', 'transmission', 'drive', 'type', 'paint_color', 'manufacturer', 'state']
normalization_cols = ['year', 'odometer']

# Definindo as etapas do pipeline
steps = ColumnTransformer([
    ('onehot', OneHotEncoder(drop='first',sparse=False), onehot_cols),
    ('normalize', MinMaxScaler(), normalization_cols)
], remainder='passthrough')

preprocessed_data = steps.fit_transform(X)
scaler = MinMaxScaler()
scaler.fit(y.values.reshape(-1, 1))

######################## INTERFACE DO USUARIO ################
def main(input_data):


    # Chamar o fit_transform para ajustar o transformador de colunas aos dados
    preprocessed_data = steps.fit_transform(input_data)

    # Código do Streamlit
    st.title('Previsão de preços de carros usados')
    
    image = Image.open("C:/Users/SAMSUNG/OneDrive/Documentos/GitHub/vendas_carros/imagens/carros_usados.webp")
    st.image(image, caption='Carros usados são um amplo mercado para quem quer um bom veículo por um preço mais razoável')
    
    st.markdown("""Estamos passando por um período de carros novos com preços muito elevados. O carro semi-novo se apresenta como uma solução para você que quer um veículo mas não tem os recursos para um zerado.
Com este aplicativo é possível escolher as características do carro desejado e ter a previsão de seu custo! Utilizando algoritmos de machine learning é possível prever preços e tomar decisões mais informadas para que você faça a melhor compra! """)

    st.title('Informe as características do carro desejado e iremos prever seu preço:')
    # Categorias numéricas
    year = st.text_input('Iforme o ano do veículo desejado:')
    if not is_numeric(year):
        st.warning("Por favor, insira um valor numérico para o ano.")

    odometer = st.text_input("Informe o valor do hodômetro:")

    if not is_numeric(odometer):
        st.warning("Por favor, insira um valor numérico para o hodômetro.")


    ##################         Variáveis categóricas    ########################################

    manufacturer = st.selectbox('Selecione o o fabricante:', X['manufacturer'].unique())

    # condition

    condition = st.selectbox('Selecione a condição desejada:', X['condition'].unique())


    #cylinders

    cylinders = st.selectbox('Selecione o número de cilindros desejado:',X['cylinders'].unique())

    #fuel

    fuel = st.selectbox('Selecione o tipo de combustível desejado:', X['fuel'].unique())

            
    # title_status

    title_status =  st.selectbox('Selecione o status de título desejado:', X['title_status'].unique())


    # transmission

    transmission = st.selectbox('Selecione o tipo de transmissão desejado:', X['transmission'].unique())


    #drive
    drive = st.selectbox('Selecione o tipo de tração desejado:', X['drive'].unique())

    #type 
    type = st.selectbox('Selecione o tipo de veículo desejado:', X['type'].unique())

    #paint_color
    paint_color = st.selectbox('Selecione a cor do carro desejada:', X['paint_color'].unique())

            
    #state
    state = st.selectbox('Selecione a região dos Estados Unidos desejado:', X['state'].unique())
    
    

    # Botão de previsão
    ok = st.button('Calcule o preço de seu carro seminovo')
    if ok:
        year = int(year)  
        odometer = float(odometer)
        Z = np.array([[year, manufacturer, condition, cylinders, fuel, odometer, title_status, transmission, drive, type, paint_color, state]])
        Z_df = pd.DataFrame(Z, columns=['year', 'manufacturer', 'condition', 'cylinders', 'fuel', 'odometer', 'title_status', 'transmission', 'drive', 'type', 'paint_color', 'state'])
        preprocessed_data  = steps.transform(Z_df)
        pred  = model.predict(preprocessed_data)
       # Reverter a transformação do preço
        y_pred_inverse = scaler.inverse_transform(pred.reshape(-1, 1))

        # Exibir a previsão de preço sem a transformação
        st.subheader(f'Previsão de preço: $ {y_pred_inverse[0][0]:.2f}')

        
        



# Executar o aplicativo
if __name__ == '__main__':
    main(X)

