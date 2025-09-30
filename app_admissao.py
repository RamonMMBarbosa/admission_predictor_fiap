## BIBLIOTECAS -----------------------------------------------------------------------------------------------------------------

import pandas       as pd
import numpy        as np
import streamlit    as st
import random
import json

from datetime   import date
from unidecode  import unidecode

from sklearn.pipeline import Pipeline

import joblib
from joblib import load

from classes_app_admissao import MMScaler, OHEncoder, FeatureNameCleaner, DataImputer, MatchesCreator, FeatureCreator

st.set_page_config(
    page_title='Preditor de Admissão',
    page_icon='🤖',
    layout='wide'
)

## CARREGANDO O MODELO ---------------------------------------------------------------------------------------------------------

@st.cache_resource
def load_model():
    pipeline = joblib.load('ml_model_admissao.joblib')
    return pipeline

PipelineFinal = load_model()

## CARREGANDO DADOS ------------------------------------------------------------------------------------------------------------

@st.cache_data
def load_data():

    df = pd.read_csv('https://raw.githubusercontent.com/RamonMMBarbosa/admission_predictor_fiap/refs/heads/main/admission_base_de_dados.csv', sep=',')
    df_vagas = pd.read_csv('https://raw.githubusercontent.com/RamonMMBarbosa/admission_predictor_fiap/refs/heads/main/vagas.csv', sep=',')
    

    dict_nivel_profissional = {'Aprendiz': 1, 'Trainee': 2, 'Auxiliar': 3, 'Assistente': 4, 'Técnico de Nível Médio': 5, 'Júnior': 6, 'Analista': 7, 'Pleno': 8, 'Sênior': 9, 'Especialista': 10, 'Líder': 11, 'Supervisor': 12, 'Coordenador': 13, 'Gerente': 14}
    dict_nivel_escolaridade = {'Ensino Fundamental Incompleto': 1, 'Ensino Fundamental Completo': 2, 'Ensino Médio Incompleto': 3, 'Ensino Médio Completo': 4, 'Ensino Técnico Incompleto': 5, 'Ensino Técnico Cursando': 6, 'Ensino Técnico Completo': 7, 'Ensino Superior Incompleto': 8, 'Ensino Superior Cursando': 9, 'Ensino Superior Completo': 10, 'Pós Graduação Incompleto': 11, 'Pós Graduação Cursando': 12, 'Pós Graduação Completo': 13, 'Mestrado Incompleto': 14, 'Mestrado Cursando': 15, 'Mestrado Completo': 16, 'Doutorado Incompleto': 17, 'Doutorado Cursando': 18, 'Doutorado Completo': 19}
    dict_nivel_linguagem = {'Nenhum': 0, 'Básico': 1, 'Intermediário': 2, 'Técnico': 3, 'Avançado': 4, 'Fluente': 5}

    areas_atuacao = df_vagas['area_atuacao_vaga'].dropna().unique().tolist()
    
    return df, df_vagas, dict_nivel_profissional, dict_nivel_escolaridade, dict_nivel_linguagem, areas_atuacao

## CARREGANDO DATAFRAMES -------------------------------------------------------------------------------------------------------

df, df_vagas, dict_nivel_profissional, dict_nivel_escolaridade, dict_nivel_linguagem, areas_atuacao = load_data()

pipeline_final = load_model()

## TÍTULO ----------------------------------------------------------------------------------------------------------------------

st.title('🤖 Encontre a Vaga Ideal para o seu Perfil')
st.markdown('Preencha seus dados e nossa IA irá analisar todas as nossas vagas abertas para encontrar as mais compatíveis com você.')

## FORMULÁRIO ------------------------------------------------------------------------------------------------------------------

st.header('Sobre Você')

with st.form(key='candidate_form'):

    col1, col2 = st.columns(2)

    with col1:

        data_nascimento = st.date_input(
            'Sua Data de Nascimento'
            ,min_value=date(1926, 1, 1)
            ,max_value=date.today()
            ,value=date(1990, 1, 1)
        )

        candidato_pcd = st.radio(
            'Você é uma Pessoa com Deficiência (PCD)?', ['Não', 'Sim']
            ,horizontal=True
        )

        area_atuacao_candidato = st.selectbox(
            'Sua Área de Atuação Principal'
            ,options=areas_atuacao
        )

        nivel_academico_candidato = st.selectbox(
            'Seu Nível Acadêmico'
            ,options=list(dict_nivel_escolaridade.keys())
        )

        nivel_profissional_candidato = st.selectbox(
            'Seu Nível Profissional Atual'
            ,options=list(dict_nivel_profissional.keys())
        )
        
        flg_candidato_empregado = st.radio(
            'Você está empregado atualmente?'
            ,['Não', 'Sim']
            ,horizontal=True
        )

        cargo_atual_candidato = st.text_input('Seu Cargo Atual', 'Ex: Analista de Dados')

    with col2:

        nivel_ingles_candidato = st.selectbox(
            'Seu Nível de Inglês'
            ,options=list(dict_nivel_linguagem.keys())
        )

        nivel_espanhol_candidato = st.selectbox(
            'Seu Nível de Espanhol'
            ,options=list(dict_nivel_linguagem.keys())
        )

        outro_idioma_candidato = st.selectbox(
            'Possui outro idioma?'
            ,['Nenhum', 'Alemão', 'Francês', 'Italiano', 'Japonês', 'Mandarim', 'Russo']
        )

        
        nivel_outro_idioma_candidato = st.selectbox(
            f'Nível de {outro_idioma_candidato}'
            ,options=list(dict_nivel_linguagem.keys())
        )
        
        flg_certificacao_candidato = st.radio(
            'Possui certificações relevantes?'
            ,['Não', 'Sim']
            ,horizontal=True
        )

        certificacoes_candidato = st.text_input('Principal Certificação', 'Ex: PMP, ITIL')
            
        flg_outras_certificacoes_candidato = st.radio(
            'Possui outras certificações?'
            ,['Não', 'Sim']
            ,horizontal=True
        )

    submit_button = st.form_submit_button(label='Analisar Vagas Compatíveis')

## PROCESSAMENTO E PREVISÃO ----------------------------------------------------------------------------------------------------

if submit_button and pipeline_final is not None and df_vagas is not None:
        
    with st.spinner('Analisando seu perfil em nosso banco de vagas, por favor aguarde...'):

        idade_candidato = int((date.today() - data_nascimento).days / 365.25)
        
        candidato_data = {
            'idade_candidato': idade_candidato
            ,'candidato_pcd': 1 if candidato_pcd == 'Sim' else 0
            ,'area_atuacao_candidato': area_atuacao_candidato
            ,'nivel_academico_candidato': dict_nivel_escolaridade[nivel_academico_candidato]
            ,'nivel_profissional_candidato': dict_nivel_profissional[nivel_profissional_candidato]
            ,'nivel_ingles_candidato': dict_nivel_linguagem[nivel_ingles_candidato]
            ,'nivel_espanhol_candidato': dict_nivel_linguagem[nivel_espanhol_candidato]
            ,'flg_outro_idioma_candidato': 0 if outro_idioma_candidato == "Nenhum" else 1
            ,'outro_idioma_candidato': outro_idioma_candidato
            ,'nivel_outro_idioma_candidato': dict_nivel_linguagem[nivel_outro_idioma_candidato]
            ,'flg_certificacao_candidato': 1 if flg_certificacao_candidato == 'Sim' else 0
            ,'certificacoes_candidato': certificacoes_candidato
            ,'flg_outras_certificacoes_candidato': 1 if flg_outras_certificacoes_candidato == 'Sim' else 0
            ,'flg_candidato_empregado': 1 if flg_candidato_empregado == 'Sim' else 0
            ,'cargo_atual_candidato': cargo_atual_candidato
        }

        df_candidato = pd.DataFrame([candidato_data])

        df_candidato['key'] = 1

        df_vagas_temp = df_vagas.copy()

        df_vagas_temp['key'] = 1

        df_predict = pd.merge(df_candidato, df_vagas_temp, on='key').drop('key', axis=1)

        df_predict_id_vaga = df_predict['id_vaga']

        df_predict = df_predict.drop(columns={'id_vaga'}, axis=1)

        df_predict['nivel_academico_candidato_is_missing'] = 0
        df_predict['nivel_profissional_candidato_is_missing'] = 0
        df_predict['nivel_ingles_candidato_is_missing'] = 0
        df_predict['nivel_espanhol_candidato_is_missing'] = 0
        df_predict['flg_outro_idioma_candidato_nao_informado'] = 0

        predictions = pipeline_final.predict(df_predict)
        prediction_proba = pipeline_final.predict_proba(df_predict)[:, 1]

        df_vagas['match'] = predictions
        df_vagas['probabilidade'] = prediction_proba
        
        vagas_compativeis = df_vagas[df_vagas['match'] == 1].sort_values(by='probabilidade', ascending=False)
        
        st.header("Resultados da Análise")
        
        if len(vagas_compativeis) > 0:
            st.success(f"🎉 Encontramos {len(vagas_compativeis)} vaga(s) compatível(is) com seu perfil!")

            vagas_para_mostrar = vagas_compativeis[[
                'id_vaga',
                'area_atuacao_vaga',
                'nivel_profissional_vaga',
                'probabilidade'
            ]].rename(columns={
                'id_vaga': 'ID da Vaga',
                'area_atuacao_vaga': 'Área de Atuação',
                'nivel_profissional_vaga': 'Nível Profissional',
                'probabilidade': 'Compatibilidade'
            })
            
            vagas_para_mostrar['Compatibilidade'] = vagas_para_mostrar['Compatibilidade'].apply(lambda x: f"{x:.2%}")
            
            st.dataframe(vagas_para_mostrar, use_container_width=True)
            
        else:
            st.warning("😕 No momento, não encontramos vagas com alta compatibilidade para o seu perfil. Fique de olho em nossas atualizações!")