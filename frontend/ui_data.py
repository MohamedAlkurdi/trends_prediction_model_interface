import streamlit as st

def region_countries_dataframe():
    return st.dataframe(
    {
        "Region": [
            "Africa",
            "East Asia",
            "North America & Australia",
            "Europe",
        ],
        "Countries": [
            "Kenya, Nigeria, South Africa",
            "Malaysia, Philippines, Singapore",
            "USA, Canada, Australia",
            "UK, Denmark, Finland",
        ],
    }
)

def classification_categories():
    return st.json(
        {
            "Categories": "bussiness ,real estates ,technology ,science ,media and entertainment ,art ,celebrity ,sports ,environment ,health ,fashion ,travel ,food ,tragedy ,crime ,accident ,politics ,military ,education ,literature ,history ,religion"
        }
    )

def zero_shot_classification_example():
    return st.json({"Sample": "Dune is the best movie ever.", "Labels": "CINEMA, ART, MUSIC", "Scores":"0.900,0.100,0.000"})

def categories_clusters_table():
    return st.table(
        {
            "Cluster": ["Economy", "Technology and Science", "Entertainment", "Lifestyle", "Accident", "Geopolitical", "Intellectualism"],
            "Includes": ["business, real estate", "technology, science", "media and entertainment, art, celebrity, sports", "environment, health, fashion, travel, food", "tragedy, crime, accident", "politics, military", "education, literature, history, religion"],
        }
    )