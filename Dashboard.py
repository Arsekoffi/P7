import streamlit as st
import pandas as pd
import numpy as np
import requests
import math
import plotly.graph_objects as go 
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
import json
import joblib
import seaborn as sns
from streamlit_shap import st_shap
import shap
from shap import LinearExplainer
from shap.maskers import Independent
import os
shap.initjs()


#path= "C:/Users/mr_ar/Downloads/Projet+Mise+en+prod+-+home-credit-default-risk/"
application_train = pd.read_csv("train_api.csv")
application_train=application_train.drop(columns=['Unnamed: 0'],axis=1)

application_test = pd.read_csv("test_api.csv")
application_test=application_test.drop(columns=['Unnamed: 0'],axis=1)

df=pd.read_csv('df_test_api.csv')

#df=pd.read_csv(path+'test_df.csv')
df["SK_ID_CURR"]=df["SK_ID_CURR"].convert_dtypes()
df_shap=df.copy()
sk=df["SK_ID_CURR"]
df.index=sk
df.drop(columns=["SK_ID_CURR"],inplace=True)
#model = joblib.load(path+'best_model.joblib')
#df = df.to_json()
model = joblib.load("model_api.joblib")
# we do not apply SMOTE on test data
model.steps.pop(0)

# Configuration du tableau de bord :
st.set_page_config(
    page_title="Probabilité de remboursement de crédit",
    layout="wide")

# Titre :
st.markdown("<h1 style='text-align: center; color: #5A5E6B;'>Probabilité de remboursement de crédit</h1>", unsafe_allow_html=True)

# Filtre :
id_filter = st.selectbox("Entrez identifiant client", pd.unique(sk))

inputs= {'ID':int(id_filter)}
data_json=json.dumps(inputs)


#prediction de la probabilité de remboursement du crédit
if st.checkbox('Prédiction de la probabilité de remboursement du crédit'):
      res = requests.post(url ='https://api-1nb9.onrender.com/predictions',data=json.dumps(inputs))
      st.subheader(f"Reponse from API ={res.text}")
      score=res.json()['score']
      pred=res.json()['prediction']
      X=df[df.index ==int(id_filter)]
      
      if pred == 0 :
             prob, decision = st.columns(2)
             with prob:
                    st.success(f"Probabilité : {score} ")
             with decision :
             
                 st.markdown("<h2 style='text-align: center; color: #44be6e;'>Crédit accepté</h2>", unsafe_allow_html=True)

      else :
            
            prob, decision = st.columns(2)
            with prob :
                  st.error(f"Probabilité: {score} ")
            with decision :
                  st.markdown("<h2 style='text-align: center; color: #ff3d41;'>Crédit refusé</h2>", unsafe_allow_html=True)
                 
if st.checkbox(
            "Afficher les données du client qui ont le plus influencé le calcul de son score ?"):
      X=df[df.index ==int(id_filter)]
      id_=df_shap[df_shap["SK_ID_CURR"]==int(id_filter)].index.tolist()
      fig, ax = plt.subplots(figsize=(15, 15))
      #st_shap(shap.force_plot(explainer.expected_value, shap_values[id_], features=df.iloc[[id_]],link='logit'))
      number = st.slider('Sélectionner le nombre de feautures à afficher ?', 
                              2, 20, 8)
      background = Independent(df, max_samples=100)
      explainer = LinearExplainer(model['Logreg'],background)
      shap_values = explainer(df)
      st_shap(shap.summary_plot(shap_values.values[id_,:], df.iloc[id_,:],
                              max_display=number,plot_type ="bar" ))
      st_shap(shap.force_plot(explainer.expected_value, shap_values.values[id_,:], df.iloc[id_,:]))
      
if st.checkbox(
            "Feature importance globlale "):
      #X=df[df.index ==int(id_filter)]
      
      fig, ax = plt.subplots(figsize=(15, 15))
      #number = st.slider('Sélectionner le nombre de feautures à afficher ?', 
                              #2, 20, 8)
      background = Independent(df, max_samples=100)
      explainer = LinearExplainer(model['Logreg'],background)
      shap_values = explainer(df)
     
      st_shap(shap.summary_plot(shap_values, df,max_display=number))

      #st_shap(shap.force_plot(explainer.expected_value, shap_values.values, features=X))
# The user chooses the strategy (it sets the threshold accordingly)
      #st.pyplot(fig)
 #@st.cache


def graphique(df,feature, features_client, title):

      if (not (math.isnan(features_client))):
            fig = plt.figure(figsize = (10, 4))

            t0 = df.loc[df['TARGET'] == 0]
            t1 = df.loc[df['TARGET'] == 1]
            sns.kdeplot(t0[feature].dropna(), label = 'Bon client', color='g')
            sns.kdeplot(t1[feature].dropna(), label = 'Mauvais client', color='r')
            plt.axvline(float(features_client), color="blue", 
                        linestyle='--', label = 'Position Client')

            plt.title(title, fontsize='20', fontweight='bold')
            plt.legend()
            plt.show()  
            st.pyplot(fig)
      else:
            st.write("Comparaison impossible car la valeur de cette variable n'est pas renseignée (NaN)")



def feature_engineering(df):
    new_df = pd.DataFrame()
    new_df = df.copy()
    new_df['CODE_GENDER'] = df['CODE_GENDER'].apply(lambda x: 'Femme' if x == 1 else 'Homme')
    new_df['DAYS_EMPLOYED'] = df['DAYS_EMPLOYED'].apply(lambda x : -x/365.25)
    new_df['DAYS_BIRTH'] = df['DAYS_BIRTH'].apply(lambda x : int(-x/365.25))
    new_df['FLAG_OWN_CAR'] = df['FLAG_OWN_CAR'].apply(lambda x: 'Oui' if x == 1 else 'Non')
    return new_df


#les informations relatives aux clients
features_numériques_à_selectionner={
      'CNT_CHILDREN': "NB ENFANTS",
      'DAYS_EMPLOYED': "NB ANNEES EMPLOI",
      'AMT_INCOME_TOTAL': "REVENUS",
      'AMT_CREDIT': "MONTANT CREDIT", 
      'AMT_ANNUITY': "MONTANT ANNUITES",
      'EXT_SOURCE_1': "EXT_SOURCE_1",
      'EXT_SOURCE_2': "EXT_SOURCE_2",
      'EXT_SOURCE_3': "EXT_SOURCE_3",
}

features_à_selectionner = {
      'CODE_GENDER': "GENRE",
      'DAYS_BIRTH': "AGE",
      'NAME_FAMILY_STATUS': "STATUT FAMILIAL",
      'CNT_CHILDREN': "NB ENFANTS",
      'FLAG_OWN_CAR': "POSSESSION VEHICULE",
      'FLAG_OWN_REALTY': "POSSESSION BIEN IMMOBILIER",
      'NAME_EDUCATION_TYPE': "NIVEAU EDUCATION",
      'OCCUPATION_TYPE': "EMPLOI",
      'DAYS_EMPLOYED': "NB ANNEES EMPLOI",
      'AMT_INCOME_TOTAL': "REVENUS",
      'AMT_CREDIT': "MONTANT CREDIT", 
      'NAME_CONTRACT_TYPE': "TYPE DE CONTRAT",
      'AMT_ANNUITY': "MONTANT ANNUITES",
      'NAME_INCOME_TYPE': "TYPE REVENUS",
      'EXT_SOURCE_1': "EXT_SOURCE_1",
      'EXT_SOURCE_2': "EXT_SOURCE_2",
      'EXT_SOURCE_3': "EXT_SOURCE_3",

      }


default_list=\
      ["GENRE","AGE","STATUT FAMILIAL","NB ENFANTS","REVENUS","MONTANT CREDIT"]
numerical_features = [ 'DAYS_BIRTH','CNT_CHILDREN', 'DAYS_EMPLOYED', 'AMT_INCOME_TOTAL','AMT_CREDIT','AMT_ANNUITY','EXT_SOURCE_1','EXT_SOURCE_2','EXT_SOURCE_3']

rotate_label = ["NAME_FAMILY_STATUS", "NAME_EDUCATION_TYPE"]
horizontal_layout = ["OCCUPATION_TYPE", "NAME_INCOME_TYPE"]
if st.checkbox(
            "Afficher les informations relatives aux clients ?"):
      st.header('Informations relatives au client')
      df=feature_engineering(application_test)
      X=df[df["SK_ID_CURR"] ==int(id_filter)]
      with st.spinner('Chargement des informations relatives au client...'):
            personal_df = X[list(features_à_selectionner .keys())]
            personal_df.rename(columns=features_à_selectionner , inplace=True)
            filtered = st.multiselect("Choisir les informations à afficher", 
                                    options=list(personal_df.columns),
                                    default=list(default_list))
            df_info = personal_df[filtered] 
            df_info['SK_ID_CURR'] = X.index.to_list()
            df_info = df_info.set_index('SK_ID_CURR')

            st.table(df_info.astype(str).T)
            show_all_info = st.checkbox("Afficher toutes les informations (dataframe brute)")
            if (show_all_info):
                  st.dataframe(X)
            

#comparaison avec les autres clients

if st.checkbox("Comparer aux autres clients"):
      st.header('Comparaison aux autres clients')
      df=feature_engineering(application_test)
      ap_train=feature_engineering(application_train)
      X=df[df["SK_ID_CURR"] ==int(id_filter)]
      
      with st.spinner('Chargement de la comparaison liée à la variable sélectionnée'):
            var = st.selectbox("Sélectionner une variable",
                              list(features_numériques_à_selectionner.values()))
            feature = list(features_numériques_à_selectionner.keys())\
            [list(features_numériques_à_selectionner .values()).index(var)]    
            graphique(ap_train, feature, X[feature], var)
            