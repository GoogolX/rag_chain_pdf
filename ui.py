import streamlit as st
from rag_pipeline import get_answer

st.title("RAG-chain PDF chat")

if "messages" not in st.session_state:
    st.session_state.messages = []

# add chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask me anything about your document..."):
    with st.chat_message("user"):
        st.markdown(prompt)
    
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        response = get_answer(prompt)
        st.markdown(response)
    
    st.session_state.messages.append({"role": "assistant", "content": response})

