import streamlit as st
from qdrant_client import QdrantClient
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser


@st.cache_resource(show_spinner="Connecting to Qdrant and loading embeddings model...")
def get_qdrant_retriever(qdrant_url, qdrant_api_key, collection_name):
    """
    Initializes and returns a Qdrant retriever.
    """
    # Use a local embedding model
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # Initialize Qdrant client
    client = QdrantClient(
        url=qdrant_url,
        api_key=qdrant_api_key,
    )

    # Create a Qdrant vector store instance
    vector_store = QdrantVectorStore(
        client=client,
        collection_name=collection_name,
        embedding=embeddings,
    )

    # Return the retriever
    return vector_store.as_retriever()

def format_docs(docs):
    """
    Formats the retrieved documents into a single string.
    """
    return "\n\n".join(doc.page_content for doc in docs)

@st.cache_data(show_spinner="Generating answer...")
def get_rag_chain_response(_rag_chain, question):
    """
    Invokes the RAG chain and returns the answer.
    """
    return _rag_chain.invoke({"question": question})

@st.cache_resource(show_spinner="Loading AI model...")
def get_rag_chain(_retriever, google_api_key):
    """Creates and caches the LangChain RAG chain."""
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", google_api_key=google_api_key, temperature=0.3)

    template = """
    You are a helpful assistant for students. Answer the question based only on the following context from their books:
    {context}

    Question: {question}

    Answer:
    """
    prompt = PromptTemplate.from_template(template)

    setup_and_retrieval = RunnableParallel(
        context=(lambda x: x["question"]) | _retriever, question=RunnablePassthrough()
    )
    answer_chain = setup_and_retrieval | {"answer": (RunnablePassthrough.assign(context=lambda x: format_docs(x["context"])) | prompt | llm | StrOutputParser()), "docs": lambda x: x["context"]}

    return answer_chain

def main():
    """
    Main function to run the Streamlit application.
    """
    st.set_page_config(page_title="Ask Questions", page_icon="❓")
    st.title("❓ Ask Questions About Your Books")

    try:
        qdrant_url = st.secrets["QDRANT_URL"]
        qdrant_api_key = st.secrets["QDRANT_API_KEY"]
        google_api_key = st.secrets["GOOGLE_API_KEY"]
    except KeyError as e:
        st.error(f"Missing secret: {e}. Please add it to your .streamlit/secrets.toml file.")
        st.stop()

    try:
        client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
        collections_response = client.get_collections()
        collection_names = [collection.name for collection in collections_response.collections]
    except Exception as e:
        st.error(f"Failed to connect to Qdrant or fetch collections: {e}")
        st.stop()

    if not collection_names:
        st.warning("No collections found. Please go to the 'Upload Books' page to create a collection.")
        st.stop()

    selected_collection = st.selectbox("Select a Book Collection:", collection_names)

    retriever = get_qdrant_retriever(qdrant_url, qdrant_api_key, selected_collection)
    rag_chain = get_rag_chain(retriever, google_api_key)

    question = st.text_input("Ask your question:", key="user_question")

    if st.button("Get Answer", key="get_answer"):
        if question:
            try:
                response = get_rag_chain_response(rag_chain, question)
                answer = response.get("answer")
                source_docs = response.get("docs", [])

                st.success("Here is the answer:")
                st.write(answer)

                if source_docs:
                    with st.expander("Show Sources"):
                        for doc in source_docs:
                            st.write(f"**Source:** `{doc.metadata.get('source', 'N/A')}`")
                            st.write(doc.page_content)
                            st.write("---")
            except Exception as e:
                st.error("An error occurred while generating the answer.")
                st.exception(e)
        else:
            st.warning("Please enter a question.")

if __name__ == "__main__":
    main()