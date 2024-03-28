# import libraries and dependencies
import chromadb
import streamlit as st
from llama_index.llms.ollama import Ollama
from llama_index.core import PromptTemplate
from llama_index.core.settings import Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.query_engine import CitationQueryEngine
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.langchain import LangchainEmbedding
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from llama_index.core import SimpleDirectoryReader, StorageContext, VectorStoreIndex


# Initialize the session state variables
if "index_choice" not in st.session_state:
    st.session_state.collection_name = None
    st.session_state.index_choice = None
    st.session_state.index = None
    st.session_state.messages = [{"role": "assistant", "content": "Hello! how can I help you today?"}]
    st.session_state.vectordb_path = "./data/vectordb/"
    st.session_state.docs_path = "./data/docs/"
    st.session_state.ollama_model = "mistral"
    st.session_state.ollama_base_url = "http://127.0.0.1:11434"
    st.session_state.embedding_model = "sentence-transformers/all-mpnet-base-v2"

# set global settings
Settings.llm = Ollama(
    model=st.session_state.ollama_model, 
    base_url=st.session_state.ollama_base_url, 
    temperature=0, 
    request_timeout=500.0
)
Settings.embed_model = LangchainEmbedding(
    HuggingFaceEmbeddings(
        model_name=st.session_state.embedding_model
    )
)

# set page configuration
st.set_page_config(
    page_title=f"Chat with your own documents 💬🦙",
    page_icon="🦙",
    layout="wide",
    initial_sidebar_state="auto",
    menu_items=None,
)
st.header("Chat with your own documents 💬🦙📚")
st.info(
    "Gen AI Tech Workshop RAG implementation app to retrieve and chat with your data via a Streamlit app.",
    icon="ℹ️",
)

###################################################################
# Sidebar
###################################################################

# Initialize ChromaDB client
client = chromadb.PersistentClient(path=st.session_state.vectordb_path)
existing_indexes = client.list_collections()
print(f"Existing indexes: {existing_indexes}")

with st.sidebar:
    st.write("### Index Management")
    if existing_indexes:
        index_names = [index.name for index in existing_indexes]
        selected_index_name = st.radio(
            "Existing Indexes",
                key="visibility",
                options=index_names,
            )
        # selected_index_name = st.selectbox("Select an index for the chat:", index_names)
        st.session_state.index_choice = selected_index_name
    else:
        st.write("No indexes found.")
        if st.button("Create/Update Index"):
            # add a spinner till the embeddings are loaded
            with st.spinner("Building index..."):
                """Build or update an index with the given documents."""
                chroma_collection = client.get_or_create_collection("default-collection", metadata={"hnsw:space": "cosine"})
                print(f"Created/existing collection {chroma_collection}")
                vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
                storage_context = StorageContext.from_defaults(vector_store=vector_store)
                st.session_state.index = VectorStoreIndex.from_documents(
                    documents= SimpleDirectoryReader(st.session_state.docs_path).load_data(),
                    transformations=[SentenceSplitter(chunk_size=512, chunk_overlap=20)],
                    storage_context=storage_context,
                    show_progress=True
                )
                # st.session_state.index = load_index_data(client, documents, st.session_state.collection_name)
            st.success(f"Index created/updated with documents.")
            existing_indexes = client.list_collections()


###################################################################
# Chat Interface
###################################################################

# function to add messages to the chat history
def add_to_message_history(role, content):
    message = {"role": role, "content": str(content)}
    st.session_state.messages.append(
        message
    )

# load the index if it exists
if st.session_state.index_choice:
    with st.spinner("Loading index..."):
        # Retrieve the index
        chroma_collection = client.get_or_create_collection(st.session_state.index_choice, metadata={"hnsw:space": "cosine"})
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        st.session_state.index = VectorStoreIndex.from_vector_store(
            vector_store,
            show_progress=True
        )

# 
# # set up system prompt
# template = (
#     "Please provide an answer based solely on the provided sources.\n" 
#     "When referencing information from a source, cite the appropriate source(s) using their corresponding numbers.\n"
#     "Every answer should include at least one source citation.\n"
#     "Only cite a source when you are explicitly referencing it.\n"
#     "If none of the sources are helpful, you should indicate that.\n"
#     "If user is asking something that you cannot answer from the context provided, just say you don't know the answer.\n"
#     "If you can answer from the context provided, always add citations and sources to your reply.\n"
#     "If the user's question is a greeting, then respond with a greeting.\n"
#     "For example\n:"
#     "Source 1:\n"
#     "The sky is red in the evening and blue in the morning.\n"
#     "Source 2:\n"
#     "Water is wet when the sky is red.\n"
#     "Query: When is water wet?\n"
#     "Answer: Water will be wet when the sky is red [2], which occurs in the evening [1].\n"
#     "Now it's your turn. Below are several numbered sources of information:\n"
#     "---------------------\n"
#     "{context_str}"
#     "\n---------------------\n"
#     "Query: {query_str}\n"
#     "Answer:\n"
# )
# qa_template = PromptTemplate(template)



# Initialize the query engine
# Using query engine to chat with the user
# if st.session_state.index is not None:
#     chat_engine = CitationQueryEngine.from_args(
#     index=st.session_state.index,
#     similarity_top_k=3,
#     # citation_qa_template=qa_template,
#     streaming=True,
#     verbose=True,
#     )
# else:
#     st.error("No index selected or created. Please select or create an index before proceeding.")

# Using the chat engine to chat with the user
if st.session_state.index is not None:
    chat_engine = st.session_state.index.as_chat_engine(
        chat_mode="context",
        verbose=True
    )
else:
    st.error("No index selected or created. Please select or create an index before proceeding.")

# index = load_index_data(client, documents, index_name)
if st.session_state.get('index_choice'):
    
    
    # Display the prior chat messages
    for message in st.session_state.messages:  
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # Prompt for user input and save to chat history
    if prompt := st.chat_input("Your question"):  
        add_to_message_history("user", prompt)

        # Display the new question immediately after it is entered
        with st.chat_message("user"):
            st.write(prompt)

        # Generate a new response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = chat_engine.stream_chat(prompt) # for chat engine
                # response = chat_engine.query(prompt) # for query engine
                response_str = ""
                response_container = st.empty()
                for token in response.response_gen:
                    response_str += token
                    response_container.write(response_str)
                add_to_message_history("assistant", response.response) # for chat engine
                # add_to_message_history("assistant", response) # for query engine

        # Save the state of the generator
        st.session_state.response_gen = response.response_gen