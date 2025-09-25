import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pydeck as pdk

st.title("Dados Enem 2024 São Paulo 📝")
st.markdown(
    """ 
    Verifique aqui as notas da sua escola. 

    """
)

if st.button("Faça a festa!"):
    st.balloons()
dados_escolas = pd.read_csv('Análise - Tabela da lista das escolas - Detalhado.csv')
resultados = pd.read_csv('RESULTADOS_SP_SAO_PAULO_2024.csv')

dados_escolas.columns = dados_escolas.columns.str.lower().str.replace(' ', '_')
resultados.columns = resultados.columns.str.lower().str.replace(' ', '_')
#esultados.groupby

resultados = resultados.rename(columns={'co_escola': 'código_inep'})
# Agrupando por 'codigo_inep' e calculando a média das demais colunas
resultados_consolidado = resultados.groupby('código_inep').mean(numeric_only=True).reset_index()
resultados_escolas = pd.merge(resultados_consolidado,dados_escolas, on='código_inep',how='left')
resultados_escolas = resultados_escolas.dropna(subset=['latitude', 'longitude'])

# Converte latitude e longitude para float, tratando erros
resultados_escolas['latitude'] = pd.to_numeric(resultados_escolas['latitude'], errors='coerce')
resultados_escolas['longitude'] = pd.to_numeric(resultados_escolas['longitude'], errors='coerce')

st.dataframe(resultados_escolas)

import streamlit as st
import pandas as pd
import pydeck as pdk

# 🔧 Mapeamento de cores por categoria
cores_categoria = {
    'Federal': [0, 128, 255],
    'Estadual': [0, 200, 0],
    'Municipal': [255, 165, 0],
    'Privada': [255, 0, 0]
}

# 🧠 Função para atribuir cor com base na categoria
def cor_por_categoria(categoria):
    return cores_categoria.get(categoria, [100, 100, 100])  # cor padrão cinza

# 🧼 Conversão de coordenadas
resultados_escolas['latitude'] = pd.to_numeric(resultados_escolas['latitude'], errors='coerce')
resultados_escolas['longitude'] = pd.to_numeric(resultados_escolas['longitude'], errors='coerce')
resultados_escolas = resultados_escolas.dropna(subset=['latitude', 'longitude'])

# 🎨 Aplicando cores
resultados_escolas['cor'] = resultados_escolas['categoria_administrativa'].apply(cor_por_categoria)

# 🔘 Seleção de colunas para exibir no marcador
colunas_disponiveis = [col for col in resultados_escolas.columns if col not in ['latitude', 'longitude', 'cor']]
colunas_selecionadas = st.multiselect("Selecione as colunas para exibir no marcador:", colunas_disponiveis)

# 🧠 Criando o label
def gerar_label(row):
    return "<br>".join([f"<b>{col}:</b> {row[col]}" for col in colunas_selecionadas])

resultados_escolas['label'] = resultados_escolas.apply(gerar_label, axis=1)

# 🗺️ Criando o mapa
layer = pdk.Layer(
    "ScatterplotLayer",
    data=resultados_escolas,
    get_position='[longitude, latitude]',
    get_radius=100,
    get_color='cor',
    pickable=True
)

tooltip = {
    "html": "{label}",
    "style": {"backgroundColor": "white", "color": "black"}
}

view_state = pdk.ViewState(
    latitude=resultados_escolas['latitude'].mean(),
    longitude=resultados_escolas['longitude'].mean(),
    zoom=10
)

st.pydeck_chart(pdk.Deck(
    layers=[layer],
    initial_view_state=view_state,
    tooltip=tooltip
))

