import requests
import json
import os

def fetch_events_paginated_paris(total_events=500, batch_size=100):
    """
    Récupère les événements de PARIS intra-muros en utilisant la pagination.
    """
    print(f"Début de l'extraction : Récupération de {total_events} événements sur Paris...")
    
    url = "https://public.opendatasoft.com/api/explore/v2.1/catalog/datasets/evenements-publics-openagenda/records"
    all_events = []
    
    # Calcul du nombre de boucles nécessaires
    offsets = range(0, total_events, batch_size)
    
    for offset in offsets:
        print(f"-> Appel API en cours (de {offset} à {offset + batch_size})...")
        
        params = {
            "limit": batch_size,
            "offset": offset,
            "refine": "location_city:Paris",     # Le filtre est maintenant ciblé sur Paris
            "order_by": "firstdate_begin DESC"   # On repart du futur vers le passé
        }
        
        response = requests.get(url, params=params)
        
        if response.status_code == 200:
            data = response.json()
            events_batch = data.get('results', [])
            all_events.extend(events_batch)
        else:
            print(f"❌ Erreur API ({response.status_code}) lors de l'offset {offset}: {response.text}")
            break

    return all_events

def save_to_json(data, filepath='data/raw_events.json'):
    """
    Sauvegarde la liste des événements dans un fichier JSON.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
        
    print(f"✅ Succès : {len(data)} événements ont été sauvegardés dans '{filepath}'.")

if __name__ == "__main__":
    # 1. Récupération des données (On vise 500 événements pour avoir plusieurs mois)
    events = fetch_events_paginated_paris(total_events=500, batch_size=100)
    
    # 2. Sauvegarde des données brutes
    if events:
        save_to_json(events)
    else:
        print("⚠️ Aucun événement récupéré. Vérifiez votre connexion ou les paramètres API.")