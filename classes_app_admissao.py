## BIBLIOTECAS --------------------------------------------------------------------------------------------------

import numpy as np
import pandas as pd

from unidecode import unidecode

from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder
from imblearn.over_sampling import SMOTE
from sklearn.impute import KNNImputer

## MIN MAX SCALER -----------------------------------------------------------------------------------------------

class MMScaler(BaseEstimator, TransformerMixin):

    def __init__(self, ToBe_Scaled):
        self.ToBe_Scaled = ToBe_Scaled
        self.scaler = MinMaxScaler()

    def fit(self, df, y=None):
        self.scaler.fit(df[self.ToBe_Scaled])
        return self
    
    def transform(self, df):
        
        df_copy = df.copy()
        
        df_copy[self.ToBe_Scaled] = self.scaler.transform(df_copy[self.ToBe_Scaled])

        return df_copy
    
## ONE HOT ENCODER ----------------------------------------------------------------------------------------------

class OHEncoder(BaseEstimator, TransformerMixin):
    
    def __init__(self, ToBe_Encoded):
        self.ToBe_Encoded = ToBe_Encoded
        self.encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
        self.feature_names_out_ = None

    def fit(self, df, y=None):
        self.encoder.fit(df[self.ToBe_Encoded])
        self.feature_names_out_ = self.encoder.get_feature_names_out(self.ToBe_Encoded)
        
        return self

    def transform(self, df):
        
        df_copy = df.copy()
        
        encoded_data = self.encoder.transform(df_copy[self.ToBe_Encoded])
        
        df_encoded = pd.DataFrame(
            encoded_data
            ,columns=self.feature_names_out_
            ,index=df_copy.index
        )
        
        df_copy = df_copy.drop(self.ToBe_Encoded, axis=1)
        
        df_final = pd.concat([df_copy, df_encoded], axis=1)
        
        return df_final

## FEATURE NAME CLEANER -----------------------------------------------------------------------------------------

class FeatureNameCleaner(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, df, y=None):
        return self

    def transform(self, df):
        df_copy = df.copy()
        for x in df_copy.columns:
            df_copy = df_copy.rename(columns={x: unidecode(x).replace(' ', '_').replace('.', '_').replace('/', '_').replace('-', '_').replace('[', '').replace(']', '').replace(':', '').replace('__', '_')})

        return df_copy
    
## KNN Imputer --------------------------------------------------------------------------------------------------

class DataImputer(BaseEstimator, TransformerMixin):
    
    def __init__(self, ToBe_Imputed, n_neighbors=5):
        self.ToBe_Imputed = ToBe_Imputed
        self.n_neighbors = n_neighbors
        self.imputer = KNNImputer(n_neighbors=self.n_neighbors)

    def fit(self, X, y=None):
        self.imputer.fit(X[self.ToBe_Imputed])
        return self

    def transform(self, X):
        X_copy = X.copy()
        
        imputed_subset = self.imputer.transform(X_copy[self.ToBe_Imputed])
        
        df_imputed_subset = pd.DataFrame(
            imputed_subset 
            ,columns=self.ToBe_Imputed 
            ,index=X_copy.index
        )
        
        X_copy[self.ToBe_Imputed] = df_imputed_subset
        
        return X_copy

## MATCHES CREATOR ----------------------------------------------------------------------------------------------

class MatchesCreator(BaseEstimator, TransformerMixin):
    
    def __init__(self):
        pass

    def fit(self, df, y=None):
        return self

    def transform(self, df):

        df_copy = df.copy()

        df_copy = df_copy.assign(
            match_area_atuacao = np.where(df_copy['area_atuacao_candidato'] == df_copy['area_atuacao_vaga'], 1, 0),
            match_pcd = np.where(df_copy['candidato_pcd'] == df_copy['vaga_para_pcd'], 1, 0),
            match_outro_idioma = np.where(df_copy['outro_idioma_candidato'] == df_copy['outro_idioma_vaga'], 1, 0)
        )

        for col in ['match_area_atuacao', 'match_pcd', 'match_outro_idioma']:
            df_copy[col] = df_copy[col].astype('int64')

        return df_copy
    
## FEATURE CREATOR ----------------------------------------------------------------------------------------------

class FeatureCreator(BaseEstimator, TransformerMixin):
    
    def __init__(self):
        pass

    def fit(self, df, y=None):
        return self

    def transform(self, df):

        df_copy = df.copy()

        for x in ['nivel_academico_candidato', 'nivel_profissional_candidato', 'nivel_ingles_candidato', 'nivel_espanhol_candidato']:
            if x in df_copy.columns:
                df_copy[x] = df_copy[x].round(0)

        colunas_texto = df_copy.select_dtypes(include='object').columns.tolist()

        for col in [col for col in df_copy.columns if col not in colunas_texto]:
            df_copy[col] = df_copy[col].astype('int64')
        
        return df_copy.assign(
            diff_nivel_profissional = df_copy['nivel_profissional_candidato'] - df_copy['nivel_profissional_vaga'],
            diff_nivel_academico = df_copy['nivel_academico_candidato'] - df_copy['nivel_academico_vaga'],
            diff_nivel_ingles = df_copy['nivel_ingles_candidato'] - df_copy['nivel_ingles_vaga'],
            diff_nivel_espanhol = df_copy['nivel_espanhol_candidato'] - df_copy['nivel_espanhol_vaga'],
            diff_nivel_outro_idioma = df_copy['nivel_outro_idioma_candidato'] - df_copy['nivel_outro_idioma_vaga']
        )