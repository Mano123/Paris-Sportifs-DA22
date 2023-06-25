import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

st.title('MACHINE LEARNING')

# IMPLEMENTATION DU MODELE

df=pd.read_csv('atp_after_cleaning.csv')
df=df.drop(['Unnamed: 0'],axis=1)
df.Date=pd.to_datetime(df.Date)
df=df.reset_index(drop=True)

# Suppression des colonnes inutiles

df=df.drop(['Location','Date','Round','Best of','Comment','P1sets','P2sets','Tournament','Cat_Player1','Cat_Player2','Series','Court','Surface','P1_elo','P2_elo','P1_proba_elo','P2_proba_elo','Ratio_proba','Ratio_PS','Ratio_B365'],axis=1)

# Variables explicatives
feats=df.drop('Result',axis=1)

# Variable cible
target=df.Result

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(feats,target,test_size=0.2,random_state=0)

# Separation en variable numérique et Catégorielle de test et d'entrainement

X_train_num=X_train.select_dtypes(include=['int','float'])
X_test_num=X_test.select_dtypes(include=['int','float'])
X_train_cat=X_train.select_dtypes(include='O')
X_test_cat=X_test.select_dtypes(include='O')

# Normalisation des données numériques

from sklearn.preprocessing import StandardScaler

scaler=StandardScaler()
X_train_num_scaled=pd.DataFrame(scaler.fit_transform(X_train_num),columns=X_train_num.columns)
X_test_num_scaled=pd.DataFrame(scaler.transform(X_test_num),columns=X_test_num.columns)
#y_test = y_test.reset_index(drop=True)

# Encodage variable catégorielle

from sklearn.preprocessing import OneHotEncoder

oneh=OneHotEncoder(handle_unknown='ignore')
X_train_cat_scaled=pd.DataFrame(oneh.fit_transform(X_train_cat).toarray(),columns=oneh.get_feature_names_out())
X_test_cat_scaled=pd.DataFrame(oneh.transform(X_test_cat).toarray(),columns=oneh.get_feature_names_out())

# Fusion Données enccodées

X_train_scaled=pd.concat([X_train_num_scaled,X_train_cat_scaled],axis=1)
X_test_scaled=pd.concat([X_test_num_scaled,X_test_cat_scaled],axis=1)

# Forêt aléatoire

st.header('FORÊT ALEATOIRE')

from sklearn.ensemble import RandomForestClassifier

forest=RandomForestClassifier(n_estimators=800,max_depth=4,n_jobs=-1,random_state=0)
forest.fit(X_train_scaled,y_train)

st.write('Score Forêt Aléatoire Entrainement : ',forest.score(X_train_scaled,y_train),'\n')
st.write('Score Forêt Aléatoire test : ',forest.score(X_test_scaled,y_test),'\n')

# Importance des variables du dataset

dataset_importance=pd.DataFrame({
    'Feature Name':forest.feature_names_in_,
    'Feature importance':forest.feature_importances_
}).sort_values(by='Feature importance',ascending=False).head(20).reset_index(drop=True)

fig=px.bar(dataset_importance,x='Feature importance',y='Feature Name',orientation='h')
st.write(fig)
