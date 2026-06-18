import os
from dotenv import load_dotenv

# Chargement des variables d'environnement
load_dotenv()

print("Test des imports en cours...\n")

try:
    import faiss
    print("FAISS est installé correctement.")
    
    from langchain_community.vectorstores import FAISS
    from langchain_community.embeddings import HuggingFaceEmbeddings
    print("LangChain et ses modules (FAISS, HuggingFace) sont importés avec succès.")
    
    # Syntaxe pour l'API Mistral (Version 2+)
    from mistralai.client import Mistral
    print(" Mistral (V2) est importé avec succès.")
    
    api_key = os.getenv("MISTRAL_API_KEY")
    if api_key:
        print("La clé API Mistral est bien détectée dans le fichier .env.")
        
        # Test d'initialisation du client
        client = Mistral(api_key=api_key)
        print("Le client Mistral est initialisé et prêt à l'emploi !")
    else:
        print(" Attention : La clé MISTRAL_API_KEY n'est pas trouvée dans le .env.")
        
except ImportError as e:
    print(f"Erreur d'importation : {e}")
except Exception as e:
    print(f" Une autre erreur est survenue : {e}")