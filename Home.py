import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

st.set_page_config(
    page_title="ATP PARIS SPORTIFS",
    page_icon="",
)

st.write("BIENVENUE SUR LE SITE DE PARIS SPORTIFS")

df=pd.read_csv('atp_after_construct.csv')
