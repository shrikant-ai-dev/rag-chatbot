import streamlit as st
import os
import tempfile
from qdrant_client import QdrantClient
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
import pandas as pd
from qdrant_client.http import models
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def process_and_upload_files(uploaded_files, collection_name, qdrant_url, qdrant_api_key, is_new_collection):
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
            # Load and override the source to be just the filename
            docs = loader.load()
            for doc in docs:
                doc.metadata["source"] = uploaded_file.name
            all_docs.extend(docs)

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
        force_recreate=is_new_collection,
        metadata_payload_key="metadata", # Store all metadata under this key

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
        client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
        collections_response = client.get_collections()
        existing_collections = [collection.name for collection in collections_response.collections]
    except KeyError as e:
        st.error(f"Missing secret: {e}. Please add it to your .streamlit/secrets.toml file.")
        st.stop()
    except Exception as e:
        st.error(f"Failed to connect to Qdrant: {e}")
        st.stop()

    st.header("1. Choose a Collection")

    # Let user choose between creating a new collection or adding to an existing one
    create_new_option = "<Create New Collection>"
    collection_options = [create_new_option] + existing_collections
    chosen_option = st.selectbox("Select a collection or create a new one:", collection_options)

    collection_name = ""
    is_new_collection = False
    if chosen_option == create_new_option:
        new_collection_name = st.text_input("Enter the name for the new collection:").strip()
        collection_name = new_collection_name
        is_new_collection = True
    else:
        collection_name = chosen_option

    st.header("2. Upload Your Books")
    uploaded_files = st.file_uploader(
        "Choose PDF files to add to the selected collection:",
        accept_multiple_files=True, type="pdf"
    )

    if st.button("Process and Upload Books", key="upload_books"):
        if uploaded_files and collection_name:
            with st.spinner("Processing and uploading your books... This may take a while."):
                process_and_upload_files(
                    uploaded_files, collection_name, qdrant_url, qdrant_api_key, is_new_collection
                )
            st.rerun()
        else:
            st.warning("Please select or create a collection and upload at least one file.")

    st.divider()

    st.header("Manage Existing Collections")

    if not existing_collections:
        st.info("No collections found.")
    else:
        for collection in existing_collections:
            with st.container(border=True):
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.subheader(collection)
                    
                    # Fetch all points with metadata to get book details
                    try:
                        records, _ = client.scroll(collection_name=collection, with_payload=True, limit=10000)
                        if records:
                            # Use pandas to group by source and count chunks
                            # Extract the 'metadata' field, filtering for records that have it
                            valid_metadata = [
                                rec.payload['metadata'] for rec in records
                                if rec.payload and 'metadata' in rec.payload and 'source' in rec.payload.get('metadata', {})
                            ]
                            if valid_metadata:
                                df = pd.DataFrame(valid_metadata)
                                book_details = df.groupby('source').size().reset_index(name='chunks')
                                book_details.rename(columns={'source': 'Book Name', 'chunks': 'Number of Chunks'}, inplace=True)
                                
                                # Display each book with a delete button
                                for index, row in book_details.iterrows():
                                    book_name = row["Book Name"]
                                    chunk_count = row["Number of Chunks"]
                                    book_col1, book_col2, book_col3 = st.columns([2, 1, 1])
                                    book_col1.write(f"📖 **{book_name}**")
                                    book_col2.write(f"{chunk_count} chunks")
                                    if book_col3.button("Delete Book", key=f"delete_{collection}_{book_name}", type="secondary"):
                                        client.delete_points(
                                            collection_name=collection,
                                            points_selector=models.FilterSelector(
                                                filter=models.Filter(
                                                    must=[models.FieldCondition(key="metadata.source", match=models.MatchValue(value=book_name))]
                                                )
                                            )
                                        )
                                        st.success(f"Book '{book_name}' deleted from collection '{collection}'.")
                                        st.rerun()
                            else:
                                st.write("Collection contains records, but no valid book source metadata was found.")
                        else:
                            st.write("This collection is empty.")
                    except Exception as e:
                        st.error(f"Could not retrieve details for this collection: {e}")
                with col2:
                    if st.button("Delete Collection", key=f"delete_{collection}", type="primary"):
                        client.delete_collection(collection_name=collection)
                        st.success(f"Collection '{collection}' deleted successfully.")
                        st.rerun()

if __name__ == "__main__":
    main()