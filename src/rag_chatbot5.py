import os
# Correctif pour empêcher le crash OpenMP sur Mac
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

from dotenv import load_dotenv
# Chargement de la clé API
load_dotenv()

from langchain_community.vectorstores import FAISS
from langchain_mistralai import MistralAIEmbeddings, ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

def load_rag_chain():
    """
    Initialise et retourne la chaîne RAG utilisant l'architecture LCEL pure.
    """
    print("⏳ Chargement du modèle et de la base de connaissances...")

    # 1. Le Documentaliste : Charger FAISS
    embeddings = MistralAIEmbeddings()
    vectorstore = FAISS.load_local("data/faiss_index", embeddings, allow_dangerous_deserialization=True)
    
    # On demande à FAISS de remonter les 3 meilleurs événements
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # 2. Le Cerveau : Initialiser le modèle Mistral
    llm = ChatMistralAI(model="mistral-small-latest", temperature=0.2)

    # 3. Le Prompt : La consigne stricte
    system_prompt = (
        "Tu es un guide culturel expert, chaleureux et professionnel pour la ville de Paris. "
        "Utilise UNIQUEMENT le contexte fourni ci-dessous pour recommander des événements. "
        "CONSIGNE SUR LES DATES : Si l'utilisateur mentionne une date ou une période précise (ex: juin, ce week-end), "
        "vérifie la temporalité. Si ça ne correspond pas, excuse-toi et propose l'événement trouvé comme alternative. "
        "Si l'utilisateur ne précise AUCUNE date, présente-lui simplement et avec enthousiasme les événements que tu as trouvés. "
        "Si tu ne trouves aucune information pertinente dans le contexte, dis simplement que tu n'as pas d'événement correspondant, n'invente rien.\n\n"
        "Voici les événements trouvés dans les archives :\n"
        "{context}"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    # 4. Le formatage des documents trouvés par FAISS (avec mouchard de debug)
    def format_docs(docs):
        text = "\n\n".join(doc.page_content for doc in docs)
        # --- AJOUT DU DEBUG POUR LA SOUTENANCE ---
        print("\n--- 🕵️‍♂️ CE QUE FAISS A TROUVÉ ET ENVOYÉ À MISTRAL ---")
        for i, doc in enumerate(docs):
            print(f"Doc {i+1} : {doc.metadata.get('titre', 'Sans titre')}")
        print("--------------------------------------------------\n")
        # -----------------------------------------
        return text

    # 5. Création de la chaîne LCEL
    rag_chain = (
        {"context": retriever | format_docs, "input": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain # C'est cette ligne qui avait disparu !

def chat_interface():
    """
    Interface utilisateur simple pour tester le chatbot dans le terminal.
    """
    rag_chain = load_rag_chain()
    print("\n✅ Chatbot prêt ! Posez vos questions sur les événements parisiens.")
    print("(Tapez 'exit' ou 'quit' pour arrêter)\n")

    while True:
        user_query = input("Vous 👤 : ")
        if user_query.lower() in ['exit', 'quit']:
            print("Au revoir !")
            break
            
        if not user_query.strip():
            continue

        # Exécution de la chaîne
        print("Mistral 🤖 recherche et rédige...")
        response = rag_chain.invoke(user_query) 
        
        print(f"\nMistral 🤖 : {response}\n")
        print("-" * 50)

if __name__ == "__main__":
    chat_interface()