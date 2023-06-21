import streamlit as st

st.write('Entrainement et choix du modèle')

df=pd.read_csv('atp_after_cleaning.csv')
df=df.reset_index(drop=True)
df=df.drop('Unnamed: 0',axis=1)
