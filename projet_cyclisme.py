import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import xgboost as xgb
import datetime
import folium
from joblib import dump, load
from streamlit_folium import st_folium
from pandas_geojson import read_geojson
df_clean_22 = pd.read_csv("data_22.csv", sep = ";", index_col = 0)
data_reg = pd.read_csv("data_preprop.csv", sep = ";", index_col = 0)
data_class = pd.read_csv("data_preprop.csv", sep = ";", index_col = 0)
data_class["Comptage horaire"] = pd.qcut(x = data_class["Comptage horaire"], labels = [0, 1, 2, 3], q = 4)
data_class["Comptage horaire"] = data_class["Comptage horaire"].astype("int")
df_adresses = df_clean_22[["Nom du compteur", "Lat", "Lon", "code_postal"]].drop_duplicates(subset = "Nom du compteur")

#Création structure streamlit

df=pd.read_csv("data_22.csv",sep=";",index_col=0)
pages=["Présentation", "Visualisation", "Modélisation", "Pistes de réflexion"]
page=st.sidebar.radio("Choisir une page", options=pages)

#Page des présentations
if page=="Présentation":  
    st.title(":blue[**Trafic cycliste à Paris**]")
    st.caption(" Guerchom SAID_")



#Description du dataset 

    st.subheader(":blue[Description]")
    st.write("Le projet Trafic Cycliste à Paris consiste à analyser les données fournies par la ville de Paris (open data) dans l’objectif de visualiser les horaires et les zones d’affluences de la pratique cycliste à Paris.")
    st.write("Le dataset initial contient les données qui proviennent de capteurs installées par la ville de Paris. Les données ont été récoltés entre le 29 Novembre 2021 et le 26 Janvier 2023.")
    st.write("Cependant, il a été préférable de garder uniquement les données récoltées sur l’année 2022 afin d’avoir une année civile complète à étudier.")
    

#Header
    st.write("Cliquer ci-dessous pour afficher le dataset")
    clique=st.checkbox(":blue[Cliquer]")
    if clique:
        st.write("Voici un aperçu du dataset")
        st.dataframe(df.head(5))
    st.write("Notre variable cible s’intitule :blue[*Comptage_horaire*]. Elle représente le nombre de passages de vélo par heure à une adresse donnée.")

#Les cartes en option
    Cartes=["La répartition des compteurs sur Paris","Carte des compteurs selon la fréquentation moyenne" ]
    option=st.selectbox("Choisir la carte",Cartes)

    if option==Cartes[0]:
        st.write("On constate une concentration des compteurs sur l'axe principal Est-Ouest et l'axe principal Nord-Sud.")
        st.write("On remarque que les compteurs semblent bien répartis sur tous les arrondissements, malgré des absences dans le 16ème arrondissement et au Bois de Vincennes.")
        m = folium.Map(location = (48.85341, 2.3488), zoom_start = 12)
# Import du fichier contenant les polygones d'arrondissement
        arr_json = read_geojson('arrondissements_json.geojson')
# Ajout des polygones sur la carte
        geo_map = folium.GeoJson(data=arr_json,
                         style_function=lambda feature: {'fillColor': 'yellow', 'fillOpacity' : 0.3, 'weight' : 3, 'color' : 'green'},)
        geo_map.add_to(m)
# Ajout des adresses des compteurs sur la carte
        for i, j, k, l in zip(df_adresses["Lat"], df_adresses["Lon"], df_adresses["Nom du compteur"], df_adresses["code_postal"]):
            folium.Marker(location = [i, j], icon = folium.Icon(color = "orange"), popup = k, tooltip = l).add_to(m)
        st_data=st_folium(m,width=725)

    if option==Cartes[1]:
        st.write("La carte montre qu'il y a une forte affluence au centre de Paris et surtout sur l'axe principal Est-Ouest. Les compteurs avec les faibles relevés sont situés dans le sud de Paris.")
        st.write("Sur l’ensemble de l’année 2022, les quatre arrondissements qui totalisent le plus de comptage sont les 11ème, 13ème, 15ème et 19ème arrondissements. A l’inverse, ceux qui totalisent le moins de fréquentation sont les 16ème, 17ème, 18ème et 20ème arrondissements.")
        df_ad_mean = df_clean_22.groupby(["Nom du compteur", "Lat", "Lon"]).agg({"Comptage horaire":"mean"})
        x = df_ad_mean["Comptage horaire"]
        df_ad_mean["radius"] = pd.qcut(x = x, q = 4, labels = ["20", "60", "120", "240"])
        df_ad_mean = df_ad_mean.reset_index()
#colors = ["azure", "lightblue", "skyblue", "darkblue"]
        r = folium.Map(location = (48.85341, 2.3488), zoom_start = 12)
        for i, j, k in zip(df_ad_mean["Lat"], df_ad_mean["Lon"], df_ad_mean["radius"]):
            folium.Circle(location = [i, j], radius = k, fill = True, fill_color = "blue").add_to(r)
        data_st=st_folium(r,width=725)
    
    
#VISUALISATION
if page=="Visualisation":
    st.title(":blue[**Data Visualisation**]") 
    st.markdown("Afin d'améliorer les visualisations et analyses j'ai apporté quelques modifications sur le dataset, notamment l'ajout des codes postaux.")
    st.markdown("Pour afficher le dataset après modification: cocher la case ci-dessous")
    
    Execute=st.checkbox("Afficher le dataset")
    if Execute:
        st.write("Voici les 10 premières lignes du Dataset de visualisation:")
        st.dataframe(df_clean_22.head(10))
                
    st.subheader("Graphiques et différentes analyses")

    Visual=["Comptage par code postal","Adresses les plus frequentées et les moins frequentées","Trafic Mensuel",
               "Trafic Horaire","Comptage selon la meteo","Jour ouvré contre week end","Frequentation par saison"]
    option=st.selectbox("Choisir la visualisation",Visual)

    st.write("Vous avez choisi :",option)
    if option==Visual[0]:
    
        st.image('comptage_par_arr.png')
        st.write("Analyse sur la fréquentation selon l'arrodissement:")
        st.markdown("Sur l’ensemble de l’année 2022, les quatre arrondissements qui totalisent le plus de comptage sont les 11ème, 13ème, 15ème et 19ème arrondissements. Ceux qui totalisent le moins de fréquentation sont les 16ème, 17ème, 18ème et 20ème arrondissements.")
    
    if option==Visual[1]:
    
        st.image('adresses_10.png')
        st.markdown("Analyse sur les adresses les plus et moins fréquentées:")
        st.markdown("Nous avons ensuite identifié les adresses les plus et les moins fréquentées. Les 10 compteurs ayant totalisé le plus de passages sur l’année 2022 sont situés dans les 2ème, 4ème, 10ème et 11ème arrondissements. A l’inverse, les 10 compteurs les moins fréquentés sont assez épars.")  
    
    if option==Visual[2]:
    
        st.image("par_mois.png")
        st.write("Analyse sur les mois:")
        st.markdown("On constate tout d’abord de nombreuses fluctuations du trafic selon les mois : les mois de septembre et d’octobre sont les plus fréquentés, tandis qu’il y a une forte baisse en décembre, janvier et février ainsi qu’en août.")
          
    
    if option==Visual[3]:
    
        st.image('par_heure.png')
        st.write("Analyse sur l'heure:")
        st.markdown("Une analyse selon l’heure de la journée nous permet de constater deux gros pics de fréquentation : le matin entre 8h et 10h avec des pics autour de 140 vélos par heure, et en fin de journée entre 18h et 20h avec plus de 160 vélos par heure.")
        
    
    if option==Visual[4]:

        st.image('comptage_meteo.png')
        st.write("Analyse sur la météo:")
        st.markdown("Nous avons ensuite étudié le rôle des conditions météo sur le trafic. Nous avons donc importé des données liées à la météo depuis la librairie “Meteostat”. On remarque que quand le temps est clair ou nuageux, les compteurs relèvent une très grande fréquentation qui peut aller jusqu’à plus de 1200 vélos par heure selon les compteurs. A l'inverse, quand il pleut, neige ou quand le temps est orageux, la fréquentation chute drastiquement et on obtient des relevés plus faibles, en dessous de 600 et de 200 par temps de neige et d’orage respectivement.")
        
    

    if option==Visual[5]:
        st.image('jourouvre_vs_we.png')
        st.write("Analyse sur jour ouvré contre week end:")
        st.markdown("Nous avons également comparé la fréquentation pendant les jours ouvrables à la fréquentation le week-end . On constate que le trafic cycliste est globalement plus élevé en semaine que lors du weekend. On identifie en outre des pics de fréquentation différents : les jours ouvrés, les pics de fréquentation sont ceux identifiés précédemment (8h-10h et 18h-20h) avec respectivement plus de 175 et près de 200 vélos par heure, alors que les weekends, le pic de fréquentation est aux alentours de 16h avec environ 100 vélos par heure. ")
     
    if option==Visual[6]:
        st.image('par_saison.png')
        st.write("Analyse de la fréquentation par saison:")
        st.markdown("Du côté des saisons, sans surprise, le printemps et l’été sont les saisons les plus fréquentées de l’année par les cyclistes avec respectivement 27,71% et 27,02% du trafic annuel. L’automne est aussi bien fréquenté. A l’inverse, on constate une baisse notable de la fréquentation cycliste en hiver avec moins de 20% du trafic annuel. Ces chiffres confirment l’analyse temporelle où nous avions constaté des baisses notables entre décembre et février.")
              
                   
    st.subheader(":blue[Conclusion générale sur la visualisation]")
    st.markdown("J'ai constaté le manque de corrélation importante entre les variables et la fréquentation. Néanmoins, le printemps est la saison la plus fréquentée. Certains compteurs sont très sollicités par temps de pluie et temps de neige, une attention particulière à ces axes est recommandée.Enfin, j'ai observé des pics de fréquentation entre 8h00 et 10h00, et entre 17h00 et 20h00. Notre constat est que le vélo est utilisé à Paris pour se rendre à son lieu d’activité.")
    

    
#MODELISATION
if page == "Modélisation":
    st.title(":blue[**Modélisation : prédictions du trafic cycliste**]")
    
    tab1, tab2 = st.tabs(["Modèle", "Prédictions"])
    
    with tab1:
        # Partie 1 : résumé du travail de modélisation
        st.header("Travail de modélisation")
        st.write("Ma variable cible “Comptage horaire” est une donnée numérique continue qui va de 0 à 1357. L’approche de la régression semble donc la plus pertinente de prime abord pour tenter un modèle de machine learning. Mais constatant rapidement un manque de corrélation entre les variables, j'ai aussi choisi de construire un modèle de classification. En effet, au-delà du chiffre brut du comptage horaire, il peut également être pertinent de déterminer la densité du trafic de manière qualitative (faible, moyen, élevé…)")
        
        st.subheader("1. Régression")
        st.write("Pour le modèle de régression j'ai d'abord effectué une normalisation de nos données et étudié la corrélation des variables avec une heatmap. Les quatre variables explicatives les plus intéressantes sont :")
        st.image("table_corr.jpg", width = 300)
        st.write("j'ai testé un modèle de régression linéaire **LinearRegression** qui a donné des résultats peu satisfaisants avec un score de précision R² de seulement 22%.")
        st.write("j'ai ensuite testé un modèle d'arbre de décision **DecisionTreeRegressor** qui a réussi à mieux prédire le trafic cycliste élevé avec un score de précision R² de 72%. Néanmoins il a une précision très limitée pour les valeurs élevées (après 800).")
        st.write("L'utilisation de la librairie Pycaret a permis de trouver l'algorithme le plus performant : le :blue[**Extreme Gradient Boosting**] avec un score de précision R² de 85%. Le nuage de points ci-après montre la précision de ses prédictions comparées aux valeurs réelles.")
        st.image("xgbr_graph.jpg")
        st.write("L'évaluation du modèle donne les résultats suivants :")
        st.table(pd.DataFrame(data = ["0.85", "1327.91", "1360.77", "20.54", "20.72"], index = ["R² du test", "MSE train", "MSE test" , "MAE train", "MAE test"], columns = ["Résultat"]))
        st.write("Je constate néanmoins que certaines valeurs prédites sont bien au dessus ou bien en dessous de la réalité. Le modèle a donc tendance à sur-évaluer et à sous-évaluer certaines valeurs de comptage.") 
        st.write("Les sous-évaluations sont particulièrement notables. Je constate ainsi que les prédictions du modèle sont inférieures aux données réelles pour les passages cyclistes aux heures de pointe (8h00, 9h00, 18h00 et 19h00) et dans certains arrondissements (4ème, 2ème, et 1er arrondissement.)")
        st.write("Le modèle peut donc avoir plus de mal à prédire avec précision la fréquentation élevée et dans ces arrondissements.")

        st.subheader("2. Classification")
        st.write("J'ai ensuite élaboré un modèle de classification en divisant les valeurs sur la base de nos quartiles :")
        st.table(pd.DataFrame(data = ["trafic très faible", "trafic faible", "trafic moyen", "trafic élevé"], columns = ["densité du trafic"], index = [0, 1, 2, 3]))
        st.write("J'ai d’abord testé plusieurs modèles de classification avec leurs hyper paramètres par défaut.")
        st.image("tableau_class.jpg", width = 400)
        st.write("La librairie Pycaret nous a permis d'identifier l’algorithme :blue[**Extreme Gradient Boosting**] qui donne les meilleurs résultats avec une précision sur les prédictions de l'ordre de 80%.")
        st.write("Le tableau ci-dessous montre le rapport de classification pour les 4 classes.")
        st.image("xg_class_report.png", width = 600)
        st.write("Ci-dessous la matrice de confusion du modèle qui montre la précision des prédictions sur les 4 classes.")
        st.image("xg_class_mat_conf.png", width = 600)

        st.write("A l'inverse de la régression, l'algorithme de classification 'XGBoost Classifier' est donc un peu plus performant pour prédire le trafic très faible et élevé.")
    
    with tab2:
        # Partie 2 : widget de prédiction
        st.header("Widget de prédiction")
        st.write("L'outil ci-dessous permet de sélectionner un compteur et une date pour prédire le trafic cycliste selon les deux algorithmes retenus.")
        
        # Choix de l'arrondissement
        select_data = df_clean_22
        arr_select = st.selectbox("Choix de l'arrondissment", np.sort(df_adresses["code_postal"].unique()))
        code_postal = arr_select

        # Réduction du df select_data selon ce choix
        select_data = select_data[select_data["code_postal"] == arr_select]
        #select_data.head(3)

        # Choix du compteur
        compteur_select = st.selectbox("Choix du compteur", select_data["Nom du compteur"].unique())

        # Réduction du df select_data selon le choix
        select_data = select_data[select_data["Nom du compteur"] == compteur_select]

        # Transformation du choix en valeurs Lat et Lon
        Lat = float(select_data["Lat"].values[0])
        Lon = float(select_data["Lon"].values[0])

        # Choix du mois
        dico_mois = {1 :'Janvier', 2 : 'Fevrier', 3 : 'Mars', 4: 'Avril', 5 : 'Mai', 6:'Juin', 7:'Juillet', 8:'Aout', 9:'Septembre', 10:'Octobre', 11:'Novembre', 12 :'Décembre'}
                
        liste_reduite_mois = [dico_mois[i] for i in select_data["mois"].unique()]
        mois = st.selectbox("Choix du mois", liste_reduite_mois)

        for key, val in dico_mois.items():
            if val == mois:
                mois = key
        select_data = select_data[select_data["mois"] == mois]

        # Choix du jour
        jour = st.selectbox("Choix du jour", select_data["jour"].unique())
        select_data = select_data[select_data["jour"] == jour]
        # Transformation du choix en valeurs jour_sem
        jour_sem = select_data[(select_data["mois"] == mois) & (select_data["jour"] == jour)]["jour_sem"].values[0]
        # Transformation du choix en valeurs Saisons
        dico_saisons = {"hiver" : 0, "printemps" : 1, "ete" : 2, "automne" : 3}
        valeur_saison = select_data[(select_data["mois"] == mois) & (select_data["jour"] == jour)]["Saisons"].values[0]
        Saisons = dico_saisons[valeur_saison]

        # Choix de l'heure et des conditions météo
        heure = st.selectbox("Choix de l'heure", np.sort(select_data["heure"].unique()))
        select_data = select_data[select_data["heure"] == heure]
        # Transformation du choix en valeurs météo et affichage des valeurs
        dico_coco = {"Clair" : 0, "Nuageux" : 1, "Pluie" : 2, "Neige" : 3, "Orage" : 4}
        coco_values = select_data["coco"].unique()
        coco = dico_coco[coco_values[0]]
        degrés = float(select_data["degrés"].values)
        pluie = float(select_data["pluie%"].values)
        vent = float(select_data["vent_kmh"].values)

        st.markdown("Conditions météo du jour choisi :")
        st.markdown("Temps : " + coco_values[0])
        st.markdown("Température : " + str(degrés) + "°C")
        st.markdown("Pluie : " + str(pluie) + "%")
        st.markdown("Vent : " + str(vent) + "km/h")

        #st.write("select_data") #verif
        #st.dataframe(select_data) #verif
        
        # import du scaler
        scaler = load("scaler.joblib")

        col=["mois", "jour", "jour_sem", "heure", "Saisons", "Lat", "Lon", "code_postal", "degrés", "pluie%", "vent_kmh", "coco"]
        X_select = pd.DataFrame(columns = col)
        X_select.loc[0] = [mois, jour, jour_sem, heure, Saisons, Lat, Lon, code_postal, degrés, pluie, vent, coco]
        #st.write("X_select") #verif
        #st.dataframe(X_select) #verif
                
        X_select_reduit = pd.DataFrame(scaler.transform(X_select), columns = col)
        #st.write("X_select_reduit") #verif
        #st.dataframe(X_select_reduit) #verif
        
        if st.button("Prédire", type = "primary"):

            # REGRESION
            st.subheader("**:green[Prédiction avec le modèle de régression : XGBoost Regressor]**")
            
            xgbr = load("XGBRegressor.joblib")

            # Afficher la prédiction
            pred_select = xgbr.predict(X_select_reduit)
            st.write("Trafic prédit :", np.round(pred_select[0],0))

            # Afficher la valeur réelle
            reg_reel = data_reg.loc[(data_reg["mois"] == mois) 
            & (data_reg["jour"] == jour) 
            & (data_reg["jour_sem"] == jour_sem) 
            & (data_reg["heure"] == heure) 
            & (data_reg["Saisons"] == Saisons) 
            & (data_reg["Lat"] == Lat)
            & (data_reg["Lon"] == Lon) 
            & (data_reg["code_postal"] == code_postal) 
            & (data_reg["degrés"] == degrés) 
            & (data_reg["pluie%"] == pluie) 
            & (data_reg["vent_kmh"] == vent) 
            & (data_reg["coco"] == coco)]
            #st.write("reg_reel")
            #st.dataframe(reg_reel)

            if len(reg_reel) == 1:
                valeur_reelle_reg = float(reg_reel["Comptage horaire"].values)
                st.write("Trafic réel :", valeur_reelle_reg)

            else:
                for i in range(len(reg_reel)):
                    st.write("Trafic réel :", float(reg_reel["Comptage horaire"].values[i]))
                st.write("_*Un même compteur peut avoir deux relevés réels, un pour chaque sens de circulation._")

             # CLASSIFICATION
            st.subheader("**:green[Prédiction avec le modèle de classification : XGBoost Classifier]**")
            
            xgbc = load("XGBClassifier.joblib")

            # Afficher la prédiction
            pred_select = xgbc.predict(X_select_reduit)[0]
            dico_pred = {0 : "très faible", 1 : "faible", 2 : "moyen", 3 : "élevé"}
            st.write("Trafic prédit :", dico_pred[pred_select])

            # Afficher la valeur réelle
            class_reel = data_class.loc[(data_class["mois"] == mois)
            & (data_class["jour"] == jour)
            & (data_class["jour_sem"] == jour_sem)
            & (data_class["heure"] == heure)
            & (data_class["Saisons"] == Saisons)
            & (data_class["Lat"] == Lat)
            & (data_class["Lon"] == Lon)
            & (data_class["code_postal"] == code_postal)
            & (data_class["degrés"] == degrés)
            & (data_class["pluie%"] == pluie)
            & (data_class["vent_kmh"] == vent)
            & (data_class["coco"] == coco)]
            # pour vérifier : st.dataframe(class_reel)

            # Afficher la donnée réelle
            
            if len(class_reel) == 1:
                valeur_reelle_class = float(class_reel["Comptage horaire"].values)
                st.write("Trafic réel :", dico_pred[valeur_reelle_class])

            else:
                for i in range(len(class_reel)):
                    st.write("Trafic réel* :", dico_pred[float(class_reel["Comptage horaire"].values[i])])
                st.write("_*Un même compteur peut avoir deux relevés réels, un pour chaque sens de circulation._")

        else:
            st.write("")
            
            

#PISTE DE REFLEXION
if page=="Pistes de réflexion":
    st.title(":blue[**Pistes de réflexion**]")
    st.subheader(":blue[Amélioration du dataset]")
    st.write("Fournir des données plus anciennes peut permettre de repérer les tendances sur plusieurs années, et ainsi d'améliorer les prédictions. Aussi, l’ajout des codes postaux rendrait le traitement du dataset plus rapide et efficace.")
    st.subheader(":blue[Vérification des compteurs]")
    st.write("J'ai repéré des compteurs affichant une très faible voire pas de passage comme, par exemple, la rue de la Fayette ou encore le boulevard d'Ornano. Une vérification du bon fonctionnement de ces compteurs peut être envisageable. Une analyse approfondie sur ces compteurs peut être intéressante afin de déterminer si ces pistes cyclables sont nécessaires.")
    st.subheader(":blue[Ajout de compteurs]")
    st.write("Je remarque une absence des compteurs dans des zones, comme, par exemple, au Bois de Vincennes. L’ajout des compteurs à ces emplacements peut être envisageable pour les futures analyses.")
    st.subheader(":blue[La maintenance]")
    st.write("Je recommande de veiller à la maintenance des pistes cyclable fréquemment utilisées en temps de pluies et de neige. Cela concerne au total 6 adresses.")
    col1,col2,col3=st.columns(3)
    with col1:
        st.image("compteur_neige0.png")
    with col2:
        st.image("compteur_neige1.png")
    with col3:
        st.image("compteur_neige2.png")
        
        
    col4,col5,col6= st.columns(3)
    with col4:
        st.image("compteur_neige3.png")
    with col5:
        st.image("compteur_neige4.png")
    with col6:
        st.image("compteur_neige5.png")