import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

st.title('MACHINE LEARNING ET SIMULATION')

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

# Modèle Linéaire

from sklearn.linear_model import LogisticRegression

reglog=LogisticRegression(C=1,solver='lbfgs',max_iter=1000,random_state=0)
reglog.fit(X_train_scaled,y_train)

# Arbre à décision

from sklearn.tree import DecisionTreeClassifier

dt=DecisionTreeClassifier(criterion='gini',max_depth=4,random_state=0)
dt.fit(X_train_scaled,y_train)

# Forêt aléatoire

from sklearn.ensemble import RandomForestClassifier

forest=RandomForestClassifier(n_estimators=800,max_depth=4,n_jobs=-1,random_state=0)
forest.fit(X_train_scaled,y_train)

# RESUME DES PERFORMANCES DES MODELES

st.subheader('TABLEAU RECAPITULATIF DES SCORES PAR MODELES')

performance=pd.DataFrame(
    {
        'Score Entrainement':[reglog.score(X_train_scaled,y_train),dt.score(X_train_scaled,y_train),forest.score(X_train_scaled,y_train)],
        'Score Test':[reglog.score(X_test_scaled,y_test),dt.score(X_test_scaled,y_test),forest.score(X_test_scaled,y_test)]
    },index=['Modèle Linéaire','Arbre à décision','Forêt Aléatoire']
)

st.write(performance)

# Importance des variables du dataset

st.subheader('SELECTION DES CRITERES DE PERFORMANCES DU MODELE FORÊT ALEATOIRE')

dataset_importance=pd.DataFrame({
    'Feature Name':forest.feature_names_in_,
    'Feature importance':forest.feature_importances_
}).sort_values(by='Feature importance',ascending=False).head(20).reset_index(drop=True)

fig=px.bar(dataset_importance,x='Feature importance',y='Feature Name',orientation='h')
st.write(fig)

st.subheader('SIMULATION')

uploaded_file = st.file_uploader("Choisir un fichier")
if uploaded_file is not None:
    df_test=pd.read_excel(uploaded_file)

    st.write(df_test)

    # Encodage variable numérique

    feats_scaled=pd.DataFrame(scaler.fit_transform(feats.drop(['Player1','Player2'],axis=1)),columns=feats.drop(['Player1','Player2'],axis=1).columns)
    df_test_scaled=pd.DataFrame(scaler.transform(df_test.drop(['Player1','Player2'],axis=1)),columns=df_test.drop(['Player1','Player2'],axis=1).columns)

    # Forêt aléatoire
    
    from sklearn.ensemble import RandomForestClassifier
    
    forest=RandomForestClassifier(n_estimators=800,max_depth=4,n_jobs=-1,random_state=0)
    forest.fit(feats_scaled,target)

    # Simulation sur la cagnotte

    y_pred_proba=forest.predict_proba(df_test_scaled)
    y_pred=forest.predict(df_test_scaled)
    
    capital_depart=100
    surete=0.8
    
    cagnotte=0
    
    for i,probas in enumerate(y_pred_proba):
    
      cote_player1=df_test[['P1_PS','P1_B365']].loc[i]
      cote_player2=df_test[['P2_PS','P2_B365']].loc[i]
    
      if probas[0]<surete:
        st.write('Miser {}€ sur le joueur 2 {} -'.format(round(capital_depart*probas[0]),df_test.Player2.loc[i]),'sur {}, avec une cote à {} -'.format(cote_player2.idxmax(), cote_player2.max()),'nous ferait perdre {}€'.format(round(capital_depart*probas[0])))
      elif probas[0]>surete:
        st.write('Miser {}€ sur le joueur 2 {} -'.format(round(capital_depart*probas[0]),df_test.Player2.loc[i]),'sur {}, avec une cote à {} -'.format(cote_player2.idxmax(), cote_player2.max()),'nous ferait gagner {}€'.format(round(capital_depart*probas[0]*cote_player2.max())))
        cagnotte+=round(capital_depart*probas[0]*cote_player2.max())
      elif probas[1]<surete:
        st.write('Miser {}€ sur le joueur 1 {} -'.format(round(capital_depart*probas[1]),df_test.Player2.loc[i]),'sur {}, avec une cote à {} -'.format(cote_player1.idxmax(), cote_player1.max()),'nous ferait perdre {}€'.format(round(capital_depart*probas[1])))
      elif probas[1]>surete:
        st.write('Miser {}€ sur le joueur 1 {} -'.format(round(capital_depart*probas[1]),df_test.Player2.loc[i]),'sur {}, avec une cote à {} -'.format(cote_player1.idxmax(), cote_player1.max()),'nous ferait gagner {}€'.format(round(capital_depart*probas[1]*cote_player1.max())))
        cagnotte+=round(capital_depart*probas[0]*cote_player2.max())
    
    st.write('Nous finissons la saison avec une cagnotte de {}€'.format(round(cagnotte)))
