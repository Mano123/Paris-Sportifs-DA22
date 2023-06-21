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

st.sidebar.success("Select a demo above.")

df=pd.read_csv('atp_after_construct.csv')

# NETTOYAGE DU DATASET

df.info()

# Suppression des colonnes inutiles

df=df.drop('Unnamed: 0',axis=1)
df=df.drop('ATP',axis=1)
df=df[df.Comment=='Completed']
df=df[df.P1Rank!=0]
df=df[df.P2Rank!=0]

# Gestion des valeurs manquantes

df.dropna(subset='P1sets',axis=0,how='any',inplace=True)
df.dropna(subset='P2sets',axis=0,how='any',inplace=True)

# On va aussi supprimer les lignes pour lesquelles nous n'avons aucune information sur les cotations soit environ 11% du dataset."""

df=df.dropna(subset=['P1_PS','P2_PS','P1_B365','P2_B365'],how='all',axis=0)

# On constate une forte correlation entre les cotations, ceci nous permet d'estimer les colonnes manquantes de cotation restantes"""

df.loc[:,'P1_PS'][df.P1_PS.isna()]=df.loc[:,'P1_B365'][df.P1_PS.isna()]
df.loc[:,'P2_PS'][df.P2_PS.isna()]=df.loc[:,'P2_B365'][df.P2_PS.isna()]
df.loc[:,'P1_B365'][df.P1_B365.isna()]=df.loc[:,'P1_PS'][df.P1_B365.isna()]
df.loc[:,'P2_B365'][df.P2_B365.isna()]=df.loc[:,'P2_PS'][df.P2_B365.isna()]

df.info()

# On a un dataset sans valeurs manquantes

# Gestion des doublons

df.duplicated().sum()

# Nous n'avons aucun doublons dans le dataset

## Gestion des données textuelles

# Création de deux fonctions pour supprimer les espaces à gauche et à droite pour chaque colonne variable qui lui est donnée en argument

def space_left(x):
  if x.startswith(' '):
    return x.lstrip()
  else:
    return x

def space_rigth(x):
  if x.endswith(' '):
    return x.rstrip()
  else:
    return x

df.Location=df.Location.apply(space_left)
df.Location=df.Location.apply(space_rigth)
df.Tournament=df.Tournament.apply(space_left)
df.Tournament=df.Tournament.apply(space_rigth)
df.Player1=df.Player1.apply(space_left)
df.Player1=df.Player1.apply(space_rigth)
df.Player2=df.Player2.apply(space_left)
df.Player2=df.Player2.apply(space_rigth)

# Traitement des données textuelles de la variable Location

dic_Location={
    'Ho Chi Min City':'Ho Chi Minh City','Queens Club':"Queen's Club"
}

df.Location=df.Location.replace(dic_Location)

# Traitement des données textuelles de la variable Series

dic_Series={
    'Masters Cup':'ATP Finals','Masters':'Masters 1000','International Gold':'ATP 500','ATP500':'ATP 500','International':'ATP 250','ATP250':'ATP 250'
}

df.Series=df.Series.replace(dic_Series)

# Traitement des données textuelles de la variable Tournament

dic_Tournament={
    'ATP Buenos Aires 2004':'Argentina Open','ATP Buenos Aires 2005':'Argentina Open','Copa AT&T':'Argentina Open','Copa Telmex':'Argentina Open','Copa Claro':'Argentina Open',
    'ATP Vegeta Croatia Open':'Plava Laguna Croatia Open','Konzum Croatia Open':'Plava Laguna Croatia Open','Studena Croatia Open':'Plava Laguna Croatia Open','Croatia Open':'Plava Laguna Croatia Open',
    'Abierto Mexicano':'Abierto Mexicano Telcel','Abierto Mexicano Mifel':'Abierto Mexicano Telcel','Mexican Open':'Abierto Mexicano Telcel',
    'Allianz Suisse Open':'Suisse Open Gstaad','Crédit Agricole Suisse Open Gstaad':'Suisse Open Gstaad',
    'Apia International':'Sydney International','adidas International':'Sydney International','Medibank International':'Sydney International',
    'Atlanta Tennis Championships':'Atlanta Open','BB&T Atlanta Open':'Atlanta Open','Galleryfurniture.com Tennis Challenge':'Atlanta Open',
    'BA-CA Tennis Trophy':'Erste Bank Open','CA Tennis Trophy':'Erste Bank Open',
    'Pacific Life Open':'BNP Paribas Open','Indian Wells TMS':'BNP Paribas Open',
    'Open Romania':'BRD Nastase Tiriac Trophy',
    'Chevrolet Cup':'Movistar Chile Open','Bellsouth Open':'Movistar Chile Open','Movistar Open':'Movistar Chile Open','VTR Open':'Movistar Chile Open',
    'Austrian Open':'Generali Open','Bet-At-Home Cup':'Generali Open',
    "Campionati Internazional d'Italia":"Internazionali BNL d'Italia",'Rome TMS':"Internazionali BNL d'Italia",'Telecom Italia Masters Roma':"Internazionali BNL d'Italia",
    'Synsam Swedish Open':'Swedish Open','Catella Swedish Open':'Swedish Open','SkiStar Swedish Open':'Swedish Open',
    'Gold Flake Open':'Chennai Open','Tata Open':'Chennai Open','TATA Open':'Chennai Open',
    'Citrix Tennis Championships':'Delray Beach Open','International Championships':'Delray Beach Open',
    'Colombia Open':'Claro Open Colombia',
    'Mercedes-Benz Cup':'Farmers Classic','Countrywide Classic':'Farmers Classic','LA Tennis Open':'Farmers Classic',
    'Davidoff Swiss Indoors':'Swiss Indoors Basel','Swiss Indoors':'Swiss Indoors Basel',
    'Dubai Open':'Dubai Duty Free Tennis Championships','Dubai Tennis Championships':'Dubai Duty Free Tennis Championships',"Dubai Duty Free Men's Open":'Dubai Duty Free Tennis Championships','Dubai Championships':'Dubai Duty Free Tennis Championships',
    'Energis Open':'Dutch Open','Priority Telecom Dutch Open':'Dutch Open',
    'NASDAQ-100 Open':'Miami Open','Ericsson Open':'Miami Open','Sony Ericsson Open':'Miami Open',
    'Estoril Open':'Millenium Estoril Open','Portugal Open':'Millenium Estoril Open',
    'Franklin Templeton Tennis Classic':'Arizona Tennis Classic','Channel Open':'Arizona Tennis Classic',
    'Garanti Koza Sofia Open':'Sofia Open',
    'Gazprom Hungarian Open':'Hungarian Open',
    'Geneva Open':'Gonet Geneva Open',
    'German Open Tennis Championships':'Hamburg European Open','International German Open':'Hamburg European Open','bet-at-home Open':'Hamburg European Open','German Tennis Championships':'Hamburg European Open','Hamburg TMS':'Hamburg European Open',
    'Gerry Weber Open':'Terra Wortmann Open',
    'Grand Prix de Lyon':'Open Parc Auvergne-Rhône-Alpes','Lyon Open':'Open Parc Auvergne-Rhône-Alpes',
    'Hall of Fame Championships':'Infosys Hall of Fame Open',
    'Heineken Trophy':'Libéma Open','Ordina Open':'Libéma Open','Unicef Open':'Libéma Open','Topshelf Open':'Libéma Open','Ricoh Open':'Libéma Open',
    'Idea Prokom Open':'BNP Paribas Sopot Open','Orange Prokom Open':'BNP Paribas Polish Cup',
    'Indesit ATP Milano Indoor':'Aspria Tennis Cup','Milan Indoors':'Aspria Tennis Cup','Breil ATP':'Aspria Tennis Cup','Internazionali di Lombardia':'Aspria Tennis Cup',
    'RCA Championships':'Indianapolis Tennis Championships',
    'Internationaler Raiffeisen Grand Prix':'	International Raiffeisen Grand Prix',
    'Istanbul Open':'TEB BNP Paribas Istanbul Open',
    'Japan Open':'Rakuten Japan Open Tennis Championships','AIG Japan Open Tennis Championships':'Rakuten Japan Open Tennis Championships',
    'Kremlin Cup':'VTB Kremlin Cup',
    'Kroger St. Jude':'Memphis Open','Regions Morgan Keegan Championships':'Memphis Open','U.S. National Indoor Tennis Championships':'Memphis Open',
    'Legg Mason Classic':'Citi Open',
    'Madrid Masters':'Mutua Madrid Open','Mutua Madrileña Madrid Open':'Mutua Madrid Open',
    'Mallorca Open':'Mallorca Championships',
    'Marseille Open':'Open 13 Provence','Open 13':'Open 13 Provence',
    'Mercedes Cup':'BOSS Open','Stuttgart TMS':'BOSS Open',
    'Monte Carlo Masters':'Rolex Monte-Carlo Masters',
    'Montreal TMS':'Rogers Cup','Rogers Masters':'Rogers Cup','Toronto TMS':'Rogers Cup',
    'AAPT Championships':'Adelaide International','Next Generation Hardcourts':'Adelaide International','Next Generation Adelaide International':'Adelaide International','Australian Hardcourt Championships':'Adelaide International',
    'Nottingham Open':'Nottingham Trophy','Red Letter Days Open':'Nottingham Trophy','Slazenger Open':'Nottingham Trophy','AEGON Open':'Nottingham Trophy','The Nottingham Open':'Nottingham Trophy',
    'Open Banco Sabadell':'Barcelona Open Banc Sabadell','Open Seat Godo':'Barcelona Open Banc Sabadell','Open Sabadell Atlántico 2008':'Barcelona Open Banc Sabadell',
    'Open de Moselle':'Moselle Open',
    'Qatar Open':'Qatar Exxon Mobil Open',
    'Open de Tenis Comunidad Valenciana':'Valencia Open','CAM Open Comunidad Valenciana':'Valencia Open','Valencia Open 500':'Valencia Open',
    'Pilot Pen Tennis':'Oracle Challenger Series New Haven',
    "President's Cup":'Tashkent Challenger',
    'Proton Malaysian Open':'Malaysian Open',
    'Stella Artois':'Cinch Championships','AEGON Championships':'Cinch Championships','Queens Club':'Cinch Championships',
    'Sybase Open':'SAP Open','Siebel Open':'SAP Open',
    'Heineken Open Shanghai':'Rolex Shanghai Masters','Shanghai Masters':'Rolex Shanghai Masters',
    'US Open':'New York Open','TD Waterhouse Cup':'New York Open','The Hamlet Cup':'New York Open',
    'AEGON International':'Rothesay International',
    'adidas Open':'Internationaux de Toulouse',
    'Winston-Salem Open at Wake Forest University':'Winston-Salem Open',
    'Western & Southern Financial Group Masters':'Western & Southern Open','Cincinnati TMS':'Western & Southern Open',
    'French Open':'Rolex Paris Masters','BNP Paribas':'Rolex Paris Masters','BNP Paribas Masters':'Rolex Paris Masters',
    'U.S. Clay Court Championships':"U.S. Men's Clay Court Championships",
    'Masters Cup':'ATP Finals'
}

df.loc[:,'Tournament'][df.Location=='Auckland']=df.loc[:,'Tournament'][df.Location=='Auckland'].replace({'Heineken Open':'ASB Classic'})
df.loc[:,'Tournament'][df.Location=='Shanghai']=df.loc[:,'Tournament'][df.Location=='Shanghai'].replace({'Heineken Open':'Rolex Shanghai Masters'})
df.Tournament=df.Tournament.replace(dic_Tournament)

# Traitement des données textuelles de la variable Player1

dic_Player1={
    'Al-Ghareeb M.':'Al Ghareeb M.','Andersen J.':'Andersen J.F.','Bautista R.':'Bautista Agut R.',
    'Bogomolov A.':'Bogomolov Jr.A.','Bogomolov Jr. A.':'Bogomolov Jr.A.','Carreno-Busta P.':'Carreno Busta P.',
    'Chela J.':'Chela J.I.','Del Potro J.':'Del Potro J.M.','Del Potro J. M.':'Del Potro J.M.','Dutra Da Silva R.':'Dutra Silva R.',
    'Ferrero J.':'Ferrero J.C.','Gallardo M.':'Gallardo Valles M.','Gambill J. M.':'Gambill J.M.','Gimeno D.':'Gimeno-Traver D.',
    'Gomez L.':'Gomez E.','Granollers G.':'Granollers Pujol G.','Granollers-Pujol G.':'Granollers Pujol G.','Guccione A.':'Guccione C.',
    'Haider-Mauer A.':'Haider-Maurer A.','Herbert P.H':'Herbert P.H.','Jun W.':'Jun W.S.','Kuznetsov A.':'Kuznetsov Al.','Lisnard J.':'Lisnard J.R.',
    'Marin J.A':'Marin J.A.','Mathieu P.':'Mathieu P.H.','Munoz De La Nava D.':'Munoz de la Nava D.','Munoz-De La Nava D.':'Munoz de la Nava D.',
    'Nadal R.':'Nadal-Parera R.','Qureshi A.':'Qureshi A.U.H.','Ramirez-Hidalgo R.':'Ramirez Hidalgo R.','Ramos A.':'Ramos-Vinolas A.','Riba-Madrid P.':'Riba P.',
    'Robredo R.':'Robredo T.','Schuettler P.':'Schuettler R.','Stebe C-M.':'Stebe C.M.','Van D. Merwe I.':'Van Der Merwe I.','Van der Merwe I.':'Van Der Merwe I.',
    'Viola Mat.':'Viola M.','Wang Y.T.':'Wang J.','Wang Y.':'Wang J.','van Lottum J.':'Van Lottum J.','Zhang Z.':'Zhang Ze'

}

df.Player1=df.Player1.replace(dic_Player1)

# Traitement des données textuelles de la variable Player2

dic_Player2={
    'Ancic I.':'Ancic M.','Bautista R.':'Bautista Agut R.','Bogomolov A.':'Bogomolov Jr.A.','Bogomolov Jr. A.':'Bogomolov Jr.A.','Carreno-Busta P.':'Carreno Busta P.',
    'Chela J.':'Chela J.I.','Del Potro J.':'Del Potro J.M.','Del Potro J. M.':'Del Potro J.M.','Dutra Da Silva R.':'Dutra Silva R.','Silva D.':'Dutra Silva R.','Estrella V.':'Estrella Burgos V.',
    'Ferrero J.':'Ferrero J.C.','Gambill J. M.':'Gambill J.M.','Gomez L.':'Gomez E.','Granollers G.':'Granollers Pujol G.','Granollers-Pujol G.':'Granollers Pujol G.',
    'Guccione A.':'Guccione C.','Guzman J.':'Guzman J.P.','Haider-Mauer A.':'Haider-Maurer A.','Herbert P.H':'Herbert P.H.','Kim K':'Kim K.','Kucera V.':'Kucera K.',
    'Kuznetsov A.':'Kuznetsov Al.','Levine I.':'Levine J.','Lisnard J.':'Lisnard J.R.','Marin J.A':'Marin J.A.','Mathieu P.':'Mathieu P.H.','Munoz De La Nava D.':'Munoz de la Nava D.','Munoz-De La Nava D.':'Munoz de la Nava D.',
    'Nadal R.':'Nadal-Parera R.','Nedovyesov O.':'Nedovyesov A.','Qureshi A.':'Qureshi A.U.H.','Ramirez-Hidalgo R.':'Ramirez Hidalgo R.','Ramos A.':'Ramos-Vinolas A.',
    'Riba-Madrid P.':'Riba P.','Scherrer J.':'Scherrer J.C.','Schuettler P.':'Schuettler R.',	'Statham J.':'Statham R.','Van der Merwe I.':'Van Der Merwe I.','Verdasco M.':'Verdasco F.','Vicente M.':'Vicente F.','Viola Mat.':'Viola M.',
    'Wang Y.T.':'Wang J.','Wang Y.':'Wang J.','Wang Y. Jr':'Wang Y.Jr.','Zayid M. S.':'Zayid M.S.','Zhang Z.':'Zhang Ze','van Lottum J.':'Van Lottum J.'

}

df.Player2=df.Player2.replace(dic_Player2)

# On constate qu'il y a beaucoup de catégories pour les variables Player1 et Player2, nous allons donc les catégoriser en quatre catégories différentes soit 1,2,3 et 4 selon le rang occupé par les joeurs : """

group_player1=['Professionels','Amateurs']
group_player2=['Professionels','Amateurs']
df['Cat_Player1']=pd.cut(df.P1Rank,bins=[0,200,2000],labels=group_player1)
df['Cat_Player2']=pd.cut(df.P2Rank,bins=[0,200,2000],labels=group_player2)

# On va aussi créer des colonnes supplémentaires pour les caluculer les ratios respectifs de PS, B365 et proba_elo."""

df['Ratio_PS']=df.P1_PS/df.P2_PS
df['Ratio_B365']=df.P1_B365/df.P2_B365
df['Ratio_proba']=df.P1_proba_elo/df.P2_proba_elo

df=df.reset_index(drop=True)

df.to_csv('atp_after_cleaning.csv')

st.write(df)
