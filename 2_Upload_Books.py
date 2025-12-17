import streamlit as st
import os
import tempfile
from qdrant_client import QdrantClient
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def process_and_upload_files(uploaded_files, collection_name, qdrant_url, qdrant_api_key):
    """
    Loads, splits, embeds, and uploads documents from the UI to Qdrant.
    """
    all_docs = []
    with tempfile.TemporaryDirectory() as temp_dir:
        for uploaded_file in uploaded_files:
            temp_filepath = os.path.join(temp_dir, uploaded_file.name)
            with open(temp_filepath, "wb") as f:
                f.write(uploaded_file.getvalue())
            
            st.write(f"Processing `{uploaded_file.name}`...")
            loader = PyPDFLoader(temp_filepath)
            all_docs.extend(loader.load())

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_documents(all_docs)
    st.write(f"Created {len(chunks)} chunks from {len(uploaded_files)} document(s).")

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    st.write(f"Uploading chunks to Qdrant collection: '{collection_name}'...")
    QdrantVectorStore.from_documents(
        chunks,
        embeddings,
        url=qdrant_url,
        api_key=qdrant_api_key,
        collection_name=collection_name,
        force_recreate=True,
    )
    st.success(f"✅ Successfully uploaded files to collection '{collection_name}'.")
    st.cache_data.clear()
    st.cache_resource.clear()

def main():
    """
    Main function for the Upload Books page.
    """
    st.set_page_config(page_title="Upload Books", page_icon="📤")
    st.title("📤 Upload New Books")

    try:
        qdrant_url = st.secrets["QDRANT_URL"]
        qdrant_api_key = st.secrets["QDRANT_API_KEY"]
    except KeyError as e:
        st.error(f"Missing secret: {e}. Please add it to your .streamlit/secrets.toml file.")
        st.stop()

    uploaded_files = st.file_uploader(
        "Choose PDF files", accept_multiple_files=True, type="pdf"
    )
    new_collection_name = st.text_input(
        "Enter a new collection name for these books:", key="new_collection_name"
    ).strip()

    if st.button("Create Collection", key="create_collection"):
        if uploaded_files and new_collection_name:
            with st.spinner("Processing and uploading your books... This may take a while."):
                process_and_upload_files(
                    uploaded_files, new_collection_name, qdrant_url, qdrant_api_key
                )
        else:
            st.warning("Please upload files and provide a collection name.")

if __name__ == "__main__":
    main()