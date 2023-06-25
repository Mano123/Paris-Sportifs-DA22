import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

st.title('MACHINE LEARNING')

# STATISTIQUES ET VISUALISATIONS

df=pd.read_csv('atp_after_cleaning.csv')
df=df.drop(['Unnamed: 0','Comment'],axis=1)
df.Date=pd.to_datetime(df.Date)
df=df.reset_index(drop=True)
df['Result']=df['Result'].astype('str')
