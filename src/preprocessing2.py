import pandas as pd
import json
import re
import os

def clean_html(text):
    """Supprime les balises HTML d'une chaîne de caractères."""
    if pd.isna(text):
        return "Description non disponible."
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, '', str(text))
    return re.sub(r'\s+', ' ', cleantext).strip()

def process_data_with_pandas(input_path='data/raw_events.json', output_path='data/cleaned_events.json'):
    """Nettoie et filtre les événements en utilisant Pandas."""
    print("Chargement des données brutes avec Pandas...")
    
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
    except FileNotFoundError:
        print(" Fichier introuvable.")
        return None

    # Création du DataFrame
    df = pd.DataFrame(raw_data)
    
    # 1. Sélection et renommage des colonnes utiles
    df = df[['title_fr', 'longdescription_fr', 'description_fr', 'location_name', 'location_city', 'firstdate_begin', 'lastdate_end', 'keywords_fr']]
    
    # Gestion de la description (on prend la longue, sinon la courte)
    df['description_html'] = df['longdescription_fr'].combine_first(df['description_fr'])
    
    # 2. Nettoyage du texte
    df['description'] = df['description_html'].apply(clean_html)
    
    # 3. Formatage des dates avec gestion stricte des fuseaux horaires (UTC)
    df['date_debut'] = pd.to_datetime(df['firstdate_begin'], utc=True, errors='coerce')
    
    # 4. Filtre temporel (1 an d'historique + événements à venir)
    # On utilise pd.Timestamp.now(tz='UTC') pour avoir une date de comparaison 100% compatible
    one_year_ago = pd.Timestamp.now(tz='UTC') - pd.DateOffset(years=1)
    
    # On garde les événements dont la date de début est supérieure ou égale à il y a un an
    df_filtered = df[df['date_debut'] >= one_year_ago].copy()
    
    # 5. Préparation de la structure finale
    df_filtered['lieu'] = df_filtered['location_name'].astype(str) + ", " + df_filtered['location_city'].astype(str)
    
    # Création de la colonne de texte final qui sera envoyée à Mistral pour vectorisation
    df_filtered['texte_a_vectoriser'] = "Titre: " + df_filtered['title_fr'] + " | Lieu: " + df_filtered['lieu'] + " | Description: " + df_filtered['description']
    
    # Sélection des colonnes finales
    final_cols = ['title_fr', 'description', 'lieu', 'firstdate_begin', 'lastdate_end', 'keywords_fr', 'texte_a_vectoriser']
    df_final = df_filtered[final_cols].rename(columns={'title_fr': 'titre', 'firstdate_begin': 'date_debut', 'lastdate_end': 'date_fin', 'keywords_fr': 'tags'})
    
    # Sauvegarde
    df_final.to_json(output_path, orient='records', force_ascii=False, indent=4)
    print(f" {len(df_final)} événements conservés après filtrage temporel (sauvegardés dans {output_path}).")
    
    return df_final

if __name__ == "__main__":
    process_data_with_pandas()