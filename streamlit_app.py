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
resultados_escolas = resultados_escolas.rename(columns={'nu_nota_cn': 'ciências_da_natureza','nu_nota_ch':'ciências_humanas','nu_nota_lc':'linguagens_e_códigos','nu_nota_mt':'matemática','nu_nota_redacao':'redação'})

colunas_para_converter = [
    'ciências_da_natureza',
    'ciências_humanas',
    'linguagens_e_códigos',
    'matemática',
    'redação'
]

for coluna in colunas_para_converter:
    resultados_escolas[coluna] = pd.to_numeric(resultados_escolas[coluna], errors='coerce')

colunas_para_arredondar = [
    'ciências_da_natureza',
    'ciências_humanas',
    'linguagens_e_códigos',
    'matemática',
    'redação'
]

# Arredondando para 2 casas decimais
resultados_escolas[colunas_para_arredondar] = resultados_escolas[colunas_para_arredondar].round(2)

colunas_notas = [
    'ciências_da_natureza',
    'ciências_humanas',
    'linguagens_e_códigos',
    'matemática',
    'redação'
]

# Criando a coluna 'media_geral'
resultados_escolas['média_geral'] = resultados_escolas[colunas_notas].mean(axis=1).round(2)
resultados_escolas = resultados_escolas.sort_values(by='média_geral', ascending=False)
resultados_escolas.reset_index(drop=True, inplace=True)
resultados_escolas.index += 1

resultados_escolas = resultados_escolas[['escola','dependência_administrativa','porte_da_escola','endereço','telefone','ciências_da_natureza','ciências_humanas','linguagens_e_códigos','matemática','redação','média_geral','latitude','longitude']]


st.dataframe(resultados_escolas)

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

# 🎯 Filtro por dependência administrativa
opcoes_dependencia = resultados_escolas['dependência_administrativa'].dropna().unique().tolist()
dependencia_selecionada = st.selectbox("Filtrar por dependência administrativa:", opcoes_dependencia)

# 🔍 Aplicando o filtro ao DataFrame
resultados_escolas = resultados_escolas[resultados_escolas['dependência_administrativa'] == dependencia_selecionada]



# 🎨 Aplicando cores
resultados_escolas['cor'] = resultados_escolas['dependência_administrativa'].apply(cor_por_categoria)

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

