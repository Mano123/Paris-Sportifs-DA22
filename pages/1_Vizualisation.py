import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

st.title('Data Vizualisation')

# STATISTIQUES ET VISUALISATIONS

df=pd.read_csv('atp_after_cleaning.csv')
df=df.drop(['Unnamed: 0','Comment'],axis=1)
df.Date=pd.to_datetime(df.Date)
df=df.reset_index(drop=True)
df['Result']=df['Result'].astype('str')

# Statististique sur le dataset

nums=df.select_dtypes(include=['int64','float64'])

st.header('Exploration des variables numériques')
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10, tab11, tab12= st.tabs(["P1Rank","P2Rank","P1sets","P2sets","P1_PS","P2_PS","P1_B365","P2_B365","P1_elo","P2_elo","Ratio_PS","Ratio_B365"])

tab1.subheader("P1Rank")
fig1=px.box(nums,x='P1Rank')
with tab1:
    st.plotly_chart(fig1, theme="streamlit", use_container_width=True)

tab2.subheader("P2Rank")
fig2=px.box(nums,x='P2Rank')
with tab2:
    st.plotly_chart(fig2, theme="streamlit", use_container_width=True)

tab3.subheader("P1sets")
fig3=px.box(nums,x='P1sets')
with tab3:
    st.plotly_chart(fig3, theme="streamlit", use_container_width=True)

tab4.subheader("P2sets")
fig4=px.box(nums,x='P2sets')
with tab4:
    st.plotly_chart(fig4, theme="streamlit", use_container_width=True)

tab5.subheader("P1_PS")
fig5=px.box(nums,x='P1_PS')
with tab5:
    st.plotly_chart(fig5, theme="streamlit", use_container_width=True)

tab6.subheader("P2_PS")
fig6=px.box(nums,x='P2_PS')
with tab6:
    st.plotly_chart(fig6, theme="streamlit", use_container_width=True)

tab7.subheader("P1_B365")
fig7=px.box(nums,x='P1_B365')
with tab7:
    st.plotly_chart(fig7, theme="streamlit", use_container_width=True)

tab8.subheader("P2_B365")
fig8=px.box(nums,x='P1Rank')
with tab8:
    st.plotly_chart(fig8, theme="streamlit", use_container_width=True)

tab9.subheader("P1_elo")
fig9=px.box(nums,x='P1_elo')
with tab9:
    st.plotly_chart(fig9, theme="streamlit", use_container_width=True)

tab10.subheader("P2_elo")
fig10=px.box(nums,x='P2_elo')
with tab10:
    st.plotly_chart(fig10, theme="streamlit", use_container_width=True)

tab11.subheader("Ratio_PS")
fig11=px.box(nums,x='Ratio_PS')
with tab11:
    st.plotly_chart(fig11, theme="streamlit", use_container_width=True)

tab12.subheader("Ratio_B365")
fig12=px.box(nums,x='Ratio_B365')
with tab12:
    st.plotly_chart(fig12, theme="streamlit", use_container_width=True)

cats=df.select_dtypes(include='O')

st.header('Exploration des variables catégorielles')

tab13, tab14, tab15, tab16, tab17, tab18= st.tabs(["Series","Court","Surface","Round","Cat_Player1","Cat_Player2"])

tab13.subheader('Series')
series=pd.DataFrame({'Names':cats['Series'].value_counts().index,'Values':cats['Series'].value_counts().values})
fig13=px.pie(series,values='Values',names='Names')
tab13.plotly_chart(fig13, theme="streamlit", use_container_width=True)

tab14.subheader('Court')
court=pd.DataFrame({'Names':cats['Court'].value_counts().index,'Values':cats['Court'].value_counts().values})
fig14=px.pie(court,values='Values',names='Names')
tab14.plotly_chart(fig14, theme="streamlit", use_container_width=True)

tab15.subheader('Surface')
surface=pd.DataFrame({'Names':cats['Surface'].value_counts().index,'Values':cats['Surface'].value_counts().values})
fig15=px.pie(surface,values='Values',names='Names')
tab15.plotly_chart(fig15, theme="streamlit", use_container_width=True)

tab16.subheader('Round')
round=pd.DataFrame({'Names':cats['Round'].value_counts().index,'Values':cats['Round'].value_counts().values})
fig16=px.pie(round,values='Values',names='Names')
tab16.plotly_chart(fig16, theme="streamlit", use_container_width=True)

tab17.subheader('Cat_Player1')
cat_player1=pd.DataFrame({'Names':cats['Cat_Player1'].value_counts().index,'Values':cats['Cat_Player1'].value_counts().values})
fig17=px.pie(cat_player1,values='Values',names='Names')
tab17.plotly_chart(fig17, theme="streamlit", use_container_width=True)

tab18.subheader('Cat_Player2')
cat_player2=pd.DataFrame({'Names':cats['Cat_Player2'].value_counts().index,'Values':cats['Cat_Player2'].value_counts().values})
fig18=px.pie(cat_player2,values='Values',names='Names')
tab18.plotly_chart(fig18, theme="streamlit", use_container_width=True)

st.subheader('CORRELATION ENTRE LES COTES PS et B365')
fig19=px.imshow(df[['P1_PS','P2_PS','P1_B365','P2_B365']].corr(),text_auto=True)
st.plotly_chart(fig19, theme="streamlit", use_container_width=True)