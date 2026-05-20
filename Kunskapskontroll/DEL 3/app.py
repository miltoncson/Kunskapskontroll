import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaLLM

st.title('AI-Pluggkompis')

llm = OllamaLLM(model="llama3.2")

@st.cache_resource
def setup_database():
    # Välj vilken pdf som ska användas
    loader = PyPDFLoader('Webb.pdf')
    documents = loader.load()
    
    # Delar upp texten manuellt
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_documents(documents)
    
    # Skapar vektordatabasen
    embeddings = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')
    vectorstore = Chroma.from_documents(texts, embeddings)
    return vectorstore

# Hämta databasen
db = setup_database()

# Håller koll på konversationen
if 'messages' not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    st.chat_message(message['role']).markdown(message['content'])

prompt = st.chat_input('Vad behöver du hjälp med?')

if prompt:
    st.chat_message('user').markdown(prompt)
    st.session_state.messages.append({'role': 'user', 'content': prompt})
    
    # Hittar de mest relevanta textbitarna från PDF:en baserat på frågan
    relevant_docs = db.similarity_search(prompt, k=3)
    context = "\n".join([doc.page_content for doc in relevant_docs])
    
    # Skapar prompt som ger AI:n sammanhanget
    full_prompt = f"Använd följande information för att svara på frågan:\n\n{context}\n\nFråga: {prompt}"
    
    # Skickar till Ollama
    response = llm.invoke(full_prompt)
    
    st.chat_message('assistant').markdown(response)
    st.session_state.messages.append({'role': 'assistant', 'content': response})





# ANVÄNDNING I VERKLIGHETEN
# Denna RAG-chattbot är designad för att användas som ett pedagogiskt hjälpmedel i skolan. 
# Eleven laddar upp sin lärares kursmaterial och kan sedan ställa frågor om kursen.
# Modellen svarar enbart baserat på det uppladdade materialet, vilket säkerställer att svaren är anpassade till kursens nivå och innehåll.

# MÖJLIGHETER
# - Pedagogiskt anpassad: Eftersom AI:n utgår från lärarens eget material svarar den på rätt nivå för kursen, till skillnad från
#   generella AI-verktyg som kan ge alltför avancerade svar.
# - Skalbarhet: Fungerar för alla ämnen och kurser, byt bara PDF.
# - Minskar fusk: Eleven uppmuntras att förstå sitt eget kursmaterial snarare än att få ett svar från en extern källa.

# UTMANINGAR
# - Etiskt: Det finns en risk att elever använder verktyget för att undvika att läsa kursmaterialet själva, vilket motverkar lärandet.
# - Kvalitetsberoende: Svarets kvalitet är helt beroende av att lärarens PDF-material är välskrivet och heltäckande.
# - Tekniska krav: Kräver att Ollama körs lokalt, vilket förutsätter att användaren har en dator med tillräcklig kapacitet.