import json
import os
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_mistralai import MistralAIEmbeddings

# Chargement de la clé API
load_dotenv()
api_key = os.getenv("MISTRAL_API_KEY")

def create_and_test_index(input_path='data/cleaned_events.json', persist_directory='data/faiss_index'):
    print(f" Chargement des données depuis {input_path}...")
    
    with open(input_path, 'r', encoding='utf-8') as f:
        events = json.load(f)

   # 1. Création des objets "Document" LangChain avec Métadonnées (Sécurisé)
    documents = []
    for event in events:
        texte = event.get('texte_a_vectoriser')
        
        # BOUCLIER : Si le texte est vide (None), on passe à l'événement suivant
        if not texte:
            continue
            
        doc = Document(
            page_content=str(texte), # On force le type String
            metadata={
                "titre": str(event.get('titre', 'Titre inconnu')),
                "lieu": str(event.get('lieu', 'Lieu inconnu')),
                "date_debut": str(event.get('date_debut', 'Date inconnue')),
                "tags": ", ".join(event.get('tags', [])) if isinstance(event.get('tags', []), list) else str(event.get('tags', ''))
            }
        )
        documents.append(doc)
        
    print(f"{len(documents)} documents valides préparés (les événements vides ont été ignorés).")


    # 2. Chunking (Découpage des textes)
    print(" Découpage des textes en chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,       # Taille de chaque morceau (caractères)
        chunk_overlap=50,     # Chevauchement pour ne pas couper une phrase au milieu
        separators=["\n\n", "\n", ".", " ", ""]
    )
    chunks = text_splitter.split_documents(documents)
    print(f" {len(chunks)} chunks générés à partir des documents.")

    # 3. Vectorisation et Indexation FAISS
    print(" Génération des vecteurs avec Mistral et création de l'index FAISS...")
    embeddings = MistralAIEmbeddings(api_key=api_key, model="mistral-embed")
    
    # Création de l'index (LangChain gère IndexFlatL2 automatiquement en arrière-plan)
    vectorstore = FAISS.from_documents(chunks, embeddings)
    
    # Sauvegarde locale de l'index et des métadonnées
    vectorstore.save_local(persist_directory)
    print(f" Succès : Base vectorielle complète sauvegardée dans le dossier '{persist_directory}'")

    # 4. Test de recherche sémantique (Recommandation de l'école)
    print("\n" + "="*50)
    print(" TEST DE RECHERCHE SEMANTIQUE")
    print("="*50)
    
    query = "Je cherche un concert ou de la musique en direct"
    print(f"Question : '{query}'\n")
    
    # On demande les 2 résultats les plus pertinents
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
    results = retriever.invoke(query)
    
    if results:
        for i, res in enumerate(results):
            print(f"Résultat #{i+1} : {res.metadata.get('titre')}")
            print(f"Lieu : {res.metadata.get('lieu')}")
            print(f"Extrait : {res.page_content[:150]}...\n")
    else:
        print("Aucun résultat trouvé. Essayez une autre requête.")

if __name__ == "__main__":
    create_and_test_index()