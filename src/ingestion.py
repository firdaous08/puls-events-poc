import requests
import json
import os

# S'assurer que le dossier data existe
os.makedirs('data', exist_ok=True)

def fetch_openagenda_events(limit=50):
    """
    Récupère les événements depuis l'API Open Agenda via Opendatasoft.
    """
    url = "https://public.opendatasoft.com/api/explore/v2.1/catalog/datasets/evenements-publics-openagenda/records"
    
    # Paramètres de la requête : on cible l'Île-de-France
    params = {
       "limit": limit,
        "refine": "location_region:Île-de-France",
        "order_by": "firstdate_begin DESC"
    }
    
    print(f"Récupération de {limit} événements en cours...")
    
    try:
        response = requests.get(url, params=params)
        response.raise_for_status() # Lève une erreur si la requête échoue
        
        data = response.json()
        events = data.get('results', [])
        
        # Sauvegarde des données brutes
        output_path = 'data/raw_events.json'
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(events, f, ensure_ascii=False, indent=4)
            
        print(f"✅ Succès : {len(events)} événements sauvegardés dans '{output_path}'.")
        return events
        
    except requests.exceptions.RequestException as e:
        print(f"❌ Erreur lors de l'appel API : {e}")
        return None

if __name__ == "__main__":
    # Test du script avec 50 événements pour commencer
    fetch_openagenda_events(limit=50)