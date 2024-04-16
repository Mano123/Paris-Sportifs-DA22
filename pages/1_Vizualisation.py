import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

st.header('Visualisation de données sur Power BI', divider='green')

components.iframe("https://app.powerbi.com/view?r=eyJrIjoiM2M2YjAyZjAtZjAwNC00NTAxLWIyZWYtODM2MWQzNWE3ZTNiIiwidCI6IjZiM2JlZTZlLWYxMjEtNDJkNS05ZmYxLTllOTAwMGEyOWIxMSJ9",height=600,width=1000)
