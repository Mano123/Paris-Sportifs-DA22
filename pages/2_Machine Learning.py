import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pandas.api.types import (
    is_categorical_dtype,
    is_datetime64_any_dtype,
    is_numeric_dtype,
    is_object_dtype,
)

st.title('MACHINE LEARNING ET SIMULATION')

# IMPLEMENTATION DU MODELE

df=pd.read_csv('atp_after_cleaning.csv')
df=df.drop(['Unnamed: 0'],axis=1)
df.Date=pd.to_datetime(df.Date)
df=df.reset_index(drop=True)

# Suppression des colonnes inutiles

df=df.drop(['P1sets','P2sets','Best of','Date','Comment','Ratio_proba','Ratio_PS','Ratio_B365','P1_elo','P2_elo','P1_proba_elo','P2_proba_elo'],axis=1)

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

forest=RandomForestClassifier(n_estimators=1000,max_depth=4,n_jobs=-1,random_state=0)
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

    # Encodage variable numérique

    feats_scaled=pd.DataFrame(scaler.fit_transform(feats.drop(['Location','Tournament','Series','Court','Surface','Round','Player1','Player2','Cat_Player1','Cat_Player2'],axis=1)),columns=feats.drop(['Location','Tournament','Series','Court','Surface','Round','Player1','Player2','Cat_Player1','Cat_Player2'],axis=1).columns)
    df_test_scaled=pd.DataFrame(scaler.transform(df_test.drop(['Date','Location','Tournament','Series','Court','Surface','Round','Player1','Player2'],axis=1)),columns=df_test.drop(['Date','Location','Tournament','Series','Court','Surface','Round','Player1','Player2'],axis=1).columns)

    # Forêt aléatoire
    
    from sklearn.ensemble import RandomForestClassifier
    
    forest=RandomForestClassifier(n_estimators=1000,max_depth=4,n_jobs=-1,random_state=0)
    forest.fit(feats_scaled,target)

    # Prediction de la variable cible du jeu de données de la saison 2019

    y_pred_proba=forest.predict_proba(df_test_scaled)
    y_pred=forest.predict(df_test_scaled)

    # Construction Dataframe des données prédites
    df_pred=pd.concat([df_test,pd.Series(y_pred)],axis=1)
    df_pred=df_pred.rename(columns={0:'Result'})
    
    for row in df_pred.index:
      if df_pred.loc[row,'Result']==0:
        var_echange=df_pred.loc[row,'Player1']
        df_pred.loc[row,'Player1']=df_pred.loc[row,'Player2']
        df_pred.loc[row,'Player2']=var_echange
        var_echange=df_pred.loc[row,'P1Rank']
        df_pred.loc[row,'P1Rank']=df_pred.loc[row,'P2Rank']
        df_pred.loc[row,'P2Rank']=var_echange
        var_echange=df_pred.loc[row,'P1_PS']
        df_pred.loc[row,'P1_PS']=df_pred.loc[row,'P2_PS']
        df_pred.loc[row,'P2_PS']=var_echange
        var_echange=df_pred.loc[row,'P1_B365']
        df_pred.loc[row,'P1_B365']=df_pred.loc[row,'P2_B365']
        df_pred.loc[row,'P2_B365']=var_echange

    st.title("Auto Filter Dataframes in Streamlit")
    
    
    def filter_dataframe(df: pd.DataFrame) -> pd.DataFrame:
        """
        Adds a UI on top of a dataframe to let viewers filter columns
    
        Args:
            df (pd.DataFrame): Original dataframe
    
        Returns:
            pd.DataFrame: Filtered dataframe
        """
        modify = st.checkbox("Add filters")
    
        if not modify:
            return df
    
        df = df.copy()
    
        # Try to convert datetimes into a standard format (datetime, no timezone)
        for col in df.columns:
            if is_object_dtype(df[col]):
                try:
                    df[col] = pd.to_datetime(df[col])
                except Exception:
                    pass
    
            if is_datetime64_any_dtype(df[col]):
                df[col] = df[col].dt.tz_localize(None)
    
        modification_container = st.container()
    
        with modification_container:
            to_filter_columns = st.multiselect("Filter dataframe on", df.columns)
            for column in to_filter_columns:
                left, right = st.columns((1, 20))
                left.write("↳")
            # Treat columns with < 10 unique values as categorical
                if is_categorical_dtype(df[column]) or df[column].nunique() < 10:
                    user_cat_input = right.multiselect(
                        f"Values for {column}",
                        df[column].unique(),
                        default=list(df[column].unique()),
                    )
                    df = df[df[column].isin(user_cat_input)]
                elif is_numeric_dtype(df[column]):
                    _min = float(df[column].min())
                    _max = float(df[column].max())
                    step = (_max - _min) / 100
                    user_num_input = right.slider(
                        f"Values for {column}",
                        _min,
                        _max,
                        (_min, _max),
                        step=step,
                    )
                    df = df[df[column].between(*user_num_input)]
                elif is_datetime64_any_dtype(df[column]):
                    user_date_input = right.date_input(
                        f"Values for {column}",
                        value=(
                            df[column].min(),
                            df[column].max(),
                        ),
                    )
                    if len(user_date_input) == 2:
                        user_date_input = tuple(map(pd.to_datetime, user_date_input))
                        start_date, end_date = user_date_input
                        df = df.loc[df[column].between(start_date, end_date)]
                else:
                    user_text_input = right.text_input(
                        f"Substring or regex in {column}",
                    )
                    if user_text_input:
                        df = df[df[column].str.contains(user_text_input)]
    
        return df
    
    st.dataframe(filter_dataframe(df_pred))
    

    # Simulation sur la cagnotte
    
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
