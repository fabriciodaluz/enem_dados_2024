import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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
resultados_escolas = pd.merge(resultados,dados_escolas, on='código_inep',how='left')
st.dataframe(resultados)
st.dataframe(resultados_escolas)


