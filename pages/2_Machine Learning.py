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

capital_depart = st.number_input('Capital de départ')

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
    df_pred=pd.concat([df_test,pd.Series(y_pred),pd.DataFrame(y_pred_proba,columns=['Proba 0','Proba 1'])['Proba 0'],pd.DataFrame(y_pred_proba,columns=['Proba 0','Proba 1'])['Proba 1']],axis=1)
    df_pred=df_pred.rename(columns={0:'Result',1:'Proba 0',2:'Proba 1'})

    test=[]

    modification_container = st.container()

    with modification_container:
        location = st.selectbox("Location", sorted(df_pred.Location.unique()))
        date = st.selectbox("Date",pd.to_datetime(df_pred.Date[df_pred.Location==location]).unique())
        tournament = st.selectbox("Tournament", sorted(df_pred[(df_pred.Location==location) & (df_pred.Date==date)].Tournament.unique()))
        serie = st.selectbox("Series", sorted(df_pred[(df_pred.Location==location) & (df_pred.Date==date)].Series.unique()))
        round = st.selectbox("Round", df_pred[(df_pred.Location==location) & (df_pred.Date==date)].Round.unique())
        player1 = st.selectbox("Player 1", sorted(df_pred[(df_pred.Location==location) & (df_pred.Date==date) & (df_pred.Round==round)].Player1.unique()))
        player2 = st.selectbox("Player 2", sorted(df_pred[(df_pred.Location==location) & (df_pred.Date==date) & (df_pred.Round==round) & (df_pred.Player1==player1)].Player2.unique()))
            
                
        df_filtre=df_pred[(df_pred.Location==location) & (df_pred.Tournament==tournament) & (df_pred.Series==serie) & (df_pred.Round==round) & (df_pred.Player1==player1) & (df_pred.Player2==player2)]    
    
        if st.button('FAITES VOTRE PARIS'):
            st.write('Tournoi de ',tournament)
            st.write('Date du tournoi : ',date.strftime('%d-%m-%Y'))
            st.write('Joueur 1 : ',player1)
            st.write('Probabilité que le Joueur 1 gagne : {0:.2f}'.format(df_filtre['Proba 1'].values[0]*100),' %')
            st.write('Joueur 2 : ',player2)
            st.write('Probabilité que le Joueur 2 gagne : {0:.2f}'.format(df_filtre['Proba 0'].values[0]*100),' %')
            st.write('Niveau du Tournoi : '+round)
        
            ecart=np.abs((df_filtre['Proba 1'].values[0]*100)-(df_filtre['Proba 0'].values[0]*100))
                
            gain=0
            gains=[]
            capital_actuel=capital_depart+gain
        
            if df_filtre.Result.values==1 and ecart>20:
                mise=capital_actuel*(1-df_filtre['Proba 1'].values[0])
                gain=mise*(df_filtre['P1_PS'].values[0]-1)
                st.write('Le joueur ',player1,' a plus de chance de gagner ce match par rapport au joueur ',player2)
                st.write('Miser {0:.2f}'.format(mise),' euros sur le joueur ',player1,' pour gagner {0:.2f}'.format(gain),' euros')
            elif df_filtre.Result.values==1 and ecart<20:
                mise=capital_actuel*(1-df_filtre['Proba 1'].values[0])
                gain=mise*(df_filtre['P1_PS'].values[0]-1)
                st.write("Ce paris est trop risqué, mais vous pouvez si vous le souhaitez miser {0:.2f} euros sur le joueur ".format(mise),player1," pour envisager un gain de {0:.2f} euros".format(gain))
            elif df_filtre.Result.values==0 and ecart>20:
                mise=capital_actuel*(1-df_filtre['Proba 0'].values[0])
                gain=mise*(df_filtre['P2_PS'].values[0]-1)
                st.write('Le joueur '+player2+' a plus de chance de gagner ce match par rapport au joueur '+player1+'\n'+'Miser '+str(mise)+' euros sur le joueur '+player2+' pour gagner '+str(gain)+' euros')
            elif df_filtre.Result.values==0 and ecart<20:
                mise=capital_actuel*(1-df_filtre['Proba 0'].values[0])
                gain=mise*(df_filtre['P2_PS'].values[0]-1)
                st.write("Ce paris est trop risqué, mais vous pouvez si vous le souhaitez miser {0:.2f} euros sur le joueur ".format(mise),player2," pour envisager un gain de {0:.2f} euros".format(gain))
