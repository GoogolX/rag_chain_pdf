from langchain.vectorstores import Chroma

def create_vector_db(chunks, embedding_model):
    return Chroma.from_documents(chunks, embedding_model)

