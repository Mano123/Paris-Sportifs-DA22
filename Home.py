import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

st.set_page_config(
    page_title="PROJET PARIS SPORTIFS",
    page_icon="",
)

st.title("PROJET PARIS SPORTIFS")
st.image('skysports-atp-wta-tennis-sky-sports_6369353.png')
st.header('Contexte', divider='green')
st.write("Ce projet porte sur l'ATP Tour, le principal circuit international de tennis masculin. Son équivalent féminin est le WTA Tour. Organisé par l'Association of Tennis Professionals, il a été créé en 1990 en remplacement du Grand Prix tennis circuit.")
st.write("Il est composé de tournois de plusieurs catégories au nombre desquelles nous avons :")
st.markdown("- Le Grand Chelem")
