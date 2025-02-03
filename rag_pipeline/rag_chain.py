from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from .loader import load_and_split_pdf
from .embeddings import get_embedding_model
from .vector_db import create_vector_db
from .llm import load_llm
import os

def create_qa_chain(pdf_path, chain_type="stuff", k=5, prompt=None):
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    
    print(f"Loading PDF: {pdf_path}")
    chunks = load_and_split_pdf(pdf_path)
    print(f"Created {len(chunks)} chunks from PDF")
    if len(chunks) == 0:
        raise ValueError("No chunks were created from the PDF")
    
    # adding some debug logs
    print("Getting embedding model...")
    embedding_model = get_embedding_model()
    print("Creating vector database...")
    vector_db = create_vector_db(chunks, embedding_model)
    print("Loading LLM...")
    llm = load_llm()
    print("LLM loaded successfully")

    if prompt is None:
        prompt_template = """You are a friendly and helpful assistant. Use the following pieces of context to answer the user's question in a conversational way. If you don't know the answer based on the context provided, be honest and say that you don't know, but try to be helpful by suggesting what information might be relevant.

Context: {context}

Current conversation:
{chat_history}

Question: {question}
Answer: """
        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question", "chat_history"]
        )

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key='answer'
    )

    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_db.as_retriever(search_kwargs={"k": k}),
        memory=memory,
        combine_docs_chain_kwargs={"prompt": prompt} if prompt else None,
        return_source_documents=False
    )
    return qa_chain

def get_answer(query):
    pdf_path = "Report.pdf"
    if not hasattr(get_answer, 'qa_chain'):
        get_answer.qa_chain = create_qa_chain(pdf_path)
    
    result = get_answer.qa_chain({"question": query})
    return result['answer']
