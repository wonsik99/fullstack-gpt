from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.callbacks import BaseCallbackHandler
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain_classic.embeddings import CacheBackedEmbeddings
from langchain_classic.storage import LocalFileStore
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_text_splitters import CharacterTextSplitter
import streamlit as st

st.set_page_config(page_title="PrivateGPT", page_icon="🔒")

# 1. Callback Handler
# Callback handler for displaying the LLM's streaming response on the screen in real-time
# Inherits from BaseCallbackHandler to detect LLM events (start, token generation, end)
class ChatCallbackHandler(BaseCallbackHandler):
    # Called when the LLM starts generating a response
    def on_llm_start(self, *args, **kwargs):
        self.message="" # Initialize an empty string to hold the new message
        self.message_box = st.empty() # Create an empty UI box in Streamlit (will be filled with tokens later)

    # Called when the LLM finishes generating a response
    def on_llm_end(self, *args, **kwargs):
        save_message(self.message, "ai") # Save the completed message to the session history

    # Called every time the LLM generates a token (character) (only works when streaming=True)
    def on_llm_new_token(self, token, *args, **kwargs):
        self.message += token # Append the new token to the existing message
        self.message_box.markdown(self.message) # Update the same UI box to create a typing effect

# 2. LLM
llm = ChatOllama(
    model="mistral:latest",
    temperature=0.1,
    streaming=True,
    callbacks=[
        ChatCallbackHandler()
    ]
)

# 3. Embed File
@st.cache_resource(show_spinner="Embedding file...")
def embed_file(file):
    file_content = file.read()
    file_path = f"./.cache/private_files/{file.name}"
    
    with open(file_path, "wb") as f:
        f.write(file_content)

    cache_dir = LocalFileStore(f"./.cache/private_embeddings/{file.name}")
    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n",
        chunk_size=600,
        chunk_overlap=100,
    )
    loader = UnstructuredFileLoader(file_path)
    docs = loader.load_and_split(text_splitter=splitter)
    embeddings = OllamaEmbeddings(
        model="mistral:latest"
    )
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(embeddings, cache_dir)
    vectorstore = FAISS.from_documents(docs, cached_embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 10}) #k=4 is default
    return retriever

# 4. Save Message
def save_message(message, role):
    st.session_state["messages"].append({"message":message, "role":role})

# 5. Send Message
def send_message(message, role, save=True):
    with st.chat_message(role):
        st.markdown(message)
    if save:
        save_message(message, role)

# 6. Paint History
def paint_history():
    for message in st.session_state["messages"]:
        send_message(message["message"], message["role"], save=False)

# 7. Format Docs
def format_docs(docs):
    return "\n\n".join(document.page_content for document in docs)

# 8. Prompt
prompt = ChatPromptTemplate.from_template(
    """
    Answer the question using ONLY the following context and not your training data.
    If you don't know the answer just say you don't know. DON'T make anything up.
        
    Context: {context}
    Question: {question}
    """
)

# 9. UI
st.title("PrivateGPT")

st.markdown(
    """
    Welcome!

    Use this chatbot to ask question to an AI about your files!

    Upload your files to the sidebar
    """
)

with st.sidebar:
    file = st.file_uploader(
    "Upload a .txt .pdf of .docx file",
    type=["pdf", "txt", "docx"]
    )

# 10. Chat Logic
if file:
    retriever = embed_file(file)
    send_message("I'm ready! Ask away!","ai",save=False)
    paint_history()
    message = st.chat_input("Ask anything about your file...")
    if message:
        send_message(message, "human")
        chain = (
            {
                "context": retriever | RunnableLambda(format_docs),
                "question": RunnablePassthrough(),
            }
            | prompt
            | llm
        )
        # response = chain.invoke(message)
        with st.chat_message("ai"):
            chain.invoke(message)
        
else:
    st.session_state["messages"] = []
    