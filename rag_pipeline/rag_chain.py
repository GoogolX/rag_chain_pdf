from langchain.chains import RetrievalQA
from .loader import load_and_split_pdf
from .embeddings import get_embedding_model
from .vector_db import create_vector_db
from .llm import load_llm

# Load components
pdf_path = "victor_cheng.pdf"
chunks = load_and_split_pdf(pdf_path)
embedding_model = get_embedding_model()
vector_db = create_vector_db(chunks, embedding_model)
llm = load_llm()

# Create RAG chain
qa_chain = RetrievalQA.from_chain_type(llm, retriever=vector_db.as_retriever())

def get_answer(query):
    return qa_chain.run(query)
