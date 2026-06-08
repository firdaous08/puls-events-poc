import unittest
import pandas as pd
import json
import os
from datetime import datetime, timezone
from dateutil.relativedelta import relativedelta

class TestDataPipeline(unittest.TestCase):

    def setUp(self):
        # Chemins vers les fichiers générés par nos scripts
        self.raw_path = 'data/raw_events.json'
        self.cleaned_path = 'data/cleaned_events.json'

    def test_raw_data_exists_and_is_valid(self):
        """Test 1: Vérifie que les données brutes ont bien été téléchargées"""
        self.assertTrue(os.path.exists(self.raw_path), "Le fichier raw_events.json n'existe pas.")
        with open(self.raw_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        self.assertGreater(len(data), 0, "Le fichier de données brutes est vide.")

    def test_cleaned_data_structure(self):
        """Test 2: Vérifie que Pandas a bien nettoyé et structuré les données"""
        self.assertTrue(os.path.exists(self.cleaned_path), "Le fichier cleaned_events.json n'existe pas.")
        
        df = pd.read_json(self.cleaned_path)
        
        # Vérification des colonnes attendues
        colonnes_attendues = ['titre', 'description', 'lieu', 'date_debut', 'texte_a_vectoriser']
        for col in colonnes_attendues:
            self.assertIn(col, df.columns, f"La colonne {col} est manquante dans les données nettoyées.")
            
        # Vérification qu'il n'y a plus de balises HTML dans le texte à vectoriser
        self.assertFalse(df['texte_a_vectoriser'].str.contains('<p>').any(), "Il reste des balises HTML dans les descriptions.")

    def test_time_filter(self):
        """Test 3: Vérifie que le filtre temporel (1 an) a été respecté"""
        df = pd.read_json(self.cleaned_path)
        
        # On définit la limite stricte (il y a un an en UTC)
        one_year_ago = pd.Timestamp.now(tz='UTC') - pd.DateOffset(years=1)
        
        # On s'assure que la colonne est bien en datetime UTC
        df['date_debut'] = pd.to_datetime(df['date_debut'], utc=True)
        
        # On vérifie que la date minimale du DataFrame est bien >= il y a un an
        date_minimale = df['date_debut'].min()
        self.assertGreaterEqual(date_minimale, one_year_ago, "Il y a des événements plus vieux d'un an dans le jeu de données.")

if __name__ == '__main__':
    unittest.main(verbosity=2)