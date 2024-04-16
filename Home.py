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
st.markdown("- L'ATP Finals")
st.markdown("- Le Masters 1000")
st.markdown("- L'ATP 500")
st.markdown("- L'ATP 250")
st.header('Objectifs', divider='green')
st.write("L'objectif principal de ce projet sera de battre les BookMakers (Personne morale ou physique permettant de parier de l'argent sur des évènements sportifs).")
st.write("Pour ce faire nous allons dans un premier analyser un jeu de données qui recense les matchs de Tennis joués entre 2000 et 2018 pour en tirer des visualisations et des conclusions qui pourront nous permettre dans dans la deuxième phase du projet de mettre en place un programme de Machine Learning qui pourra prédire l'issue des matchs de Tennis à jour dans le futur.")
