import streamlit as st

def main():
    """
    Main function to run the Streamlit application.
    """
    st.set_page_config(page_title="Book Chatbot", page_icon="📚")
    st.title("Welcome to the Book Chatbot! 📖")
    st.write("This application helps students get answers from their books.")
    st.sidebar.success("Select a page above.")
    st.markdown(
        """
        ### How to use this app:
        1.  Navigate to the **Upload Books** page to upload your PDF documents and create a new collection in the vector database.
        2.  Go to the **Ask Questions** page to select your book collection and start asking questions!
        """
    )

if __name__ == "__main__":
    main()
