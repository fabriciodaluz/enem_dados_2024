import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pydeck as pdk

st.title("Dados Enem 2024 Cidade de São Paulo 📝")
st.markdown(
    """ 
    Aqui voce pode checar as notas de uma determinada escola, comparar os resultados por área do conhecimento, encontrar as melhores escolas na sua região. 

    """
)
st.divider()

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

resultados_escolas = resultados_escolas[['escola','categoria_administrativa','porte_da_escola','endereço','telefone','ciências_da_natureza','ciências_humanas','linguagens_e_códigos','matemática','redação','média_geral','latitude','longitude']]


st.markdown(
    "<h4 style='text-align: left; color: #DDD; font-weight: 600;'>🔍 Encontre sua escola</h4>",
    unsafe_allow_html=True
)


# 🔍 Lista de escolas disponíveis
lista_escolas = resultados_escolas['escola'].dropna().unique().tolist()

# 🎯 Filtro interativo
escolas_selecionadas = st.multiselect("Filtrar por escola:", lista_escolas)

# 📄 Aplicando o filtro
if escolas_selecionadas:
    dados_filtrados = resultados_escolas[resultados_escolas['escola'].isin(escolas_selecionadas)]
else:
    dados_filtrados = resultados_escolas  # mostra tudo se nada for selecionado

# 📌 Exibindo o DataFrame filtrado
st.dataframe(dados_filtrados)



st.divider()

st.markdown(
    "<h4 style='text-align: left; color: #DDD; font-weight: 600;'>📊 Compare a distribuição de performance entre escolas públicas e privadas por área do conhecimento</h5>",
    unsafe_allow_html=True
)



# 🔘 Variáveis disponíveis
variaveis = [
    'ciências_da_natureza',
    'ciências_humanas',
    'linguagens_e_códigos',
    'matemática',
    'redação',
    'média_geral'
]

# 🎯 Seleção da variável
variavel_selecionada = st.selectbox("Selecione a variável para o histograma:", variaveis)

# 🧼 Garantir que os dados sejam numéricos
resultados_escolas[variavel_selecionada] = pd.to_numeric(resultados_escolas[variavel_selecionada], errors='coerce')

# 🎨 Estilo escuro e neutro
sns.set_theme(style="white")  # remove grid
plt.rcParams.update({
    'axes.facecolor': '#0e1117',
    'figure.facecolor': '#0e1117',
    'axes.edgecolor': '#444',
    'axes.labelcolor': '#DDD',
    'xtick.color': '#AAA',
    'ytick.color': '#AAA',
    'text.color': '#DDD',
    'axes.titleweight': 'semibold',
    'axes.titlesize': 14,
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans', 'Arial']
})

# 📊 Criando o histograma por categoria
fig, ax = plt.subplots()
sns.histplot(
    data=resultados_escolas,
    x=variavel_selecionada,
    hue='categoria_administrativa',
    multiple='layer',
    bins=30,
    palette='muted',
    edgecolor='#222',
    alpha=0.7,
    ax=ax
)

# 🧼 Limpeza visual
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_color('#444')
ax.spines['bottom'].set_color('#444')
ax.grid(False)

# 🏷️ Legenda sem moldura
legenda = ax.get_legend()
if legenda:
    legenda.set_frame_on(False)
    legenda.get_title().set_fontsize(12)
    legenda.get_title().set_color('#DDD')
    for text in legenda.get_texts():
        text.set_color('#DDD')

# 📌 Títulos e rótulos
ax.set_title(f'Distribuição de {variavel_selecionada} por categoria administrativa')
ax.set_xlabel(variavel_selecionada)
ax.set_ylabel('Frequência')

# 📌 Exibindo no Streamlit
st.pyplot(fig)

st.divider()


# 🏷️ Subtítulo da seção
st.markdown(
    "<h4 style='text-align: left; color: #DDD; font-weight: 600;'>📐 Compare o desempenho de duas escolas</h4>",
    unsafe_allow_html=True
)

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# 🧭 Escolas e variáveis
variaveis = [
    'ciências_da_natureza',
    'ciências_humanas',
    'linguagens_e_códigos',
    'matemática',
    'redação'
]

labels = variaveis
angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
angles += angles[:1]

# 🔘 Escolha das escolas
lista_escolas = resultados_escolas['escola'].dropna().unique().tolist()
escola_a = st.selectbox("Escolha a Escola A:", lista_escolas, key="escola_a")
escola_b = st.selectbox("Escolha a Escola B:", lista_escolas, key="escola_b")

# 🔍 Dados das escolas
dados_a = resultados_escolas[resultados_escolas['escola'] == escola_a][variaveis].mean().tolist()
dados_b = resultados_escolas[resultados_escolas['escola'] == escola_b][variaveis].mean().tolist()
dados_a += dados_a[:1]
dados_b += dados_b[:1]

# 🎨 Estilo escuro e discreto
plt.style.use("dark_background")
fig, ax = plt.subplots(figsize=(5.5, 5.5), subplot_kw=dict(polar=True))
fig.patch.set_facecolor('#0e1117')
ax.set_facecolor('#0e1117')

# 🖌️ Plotagem com cores suaves
ax.plot(angles, dados_a, label=escola_a, color='#6baed6', linewidth=2)
ax.fill(angles, dados_a, color='#6baed6', alpha=0.25)

ax.plot(angles, dados_b, label=escola_b, color='#fd8d3c', linewidth=2)
ax.fill(angles, dados_b, color='#fd8d3c', alpha=0.25)

# 🧼 Limpeza visual
ax.set_theta_offset(np.pi / 2)
ax.set_theta_direction(-1)
ax.set_rlabel_position(0)
ax.tick_params(colors='#AAA')
ax.spines['polar'].set_color('#444')
ax.grid(color='#444', linestyle='dotted', linewidth=0.5)
ax.set_xticks(angles[:-1])
ax.set_xticklabels(labels, color='#DDD', fontsize=10)
ax.set_yticklabels([])

# 🏷️ Legenda posicionada abaixo
legenda = ax.legend(
    loc='lower center',
    bbox_to_anchor=(0.5, -0.2),
    frameon=False,
    ncol=2
)
for text in legenda.get_texts():
    text.set_color('#DDD')

# 📌 Exibir no Streamlit
st.pyplot(fig)




st.divider()


st.markdown(
    "<h4 style='text-align: left; color: #DDD; font-weight: 600;'>⭐ Encontre a melhor escola da sua região</h5>",
    unsafe_allow_html=True
)

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
opcoes_dependencia = resultados_escolas['categoria_administrativa'].dropna().unique().tolist()
dependencia_selecionada = st.selectbox("Filtrar por categoria administrativa:", opcoes_dependencia)

# 🔍 Aplicando o filtro ao DataFrame
resultados_escolas = resultados_escolas[resultados_escolas['categoria_administrativa'] == dependencia_selecionada]



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

