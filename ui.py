import streamlit as st
from rag_pipeline import get_answer

st.title("Document Q&A")

query = st.text_input("Ask a question about your document:")

if st.button("Get Answer"):
    if query:
        response = get_answer(query)
        st.write("### Answer:")
        st.write(response)
    else:
        st.warning("Please enter a question.")

