from langchain_community.vectorstores import Chroma
import os

def create_vector_db(chunks, embedding_model):
    # make it persistent to prevent reprocessing pdf every time 
    # (for now, needs to be reset when we allow for uploading pdfs in UI)
    persist_directory = "chroma_db"
    if not os.path.exists(persist_directory):
        os.makedirs(persist_directory)
        return Chroma.from_documents(
            documents=chunks,
            embedding=embedding_model,
            persist_directory=persist_directory
        )
    else:
        try:
            vectordb = Chroma(
                persist_directory=persist_directory,
                embedding_function=embedding_model
            )
            if len(chunks) > 0:
                vectordb = Chroma.from_documents(
                    documents=chunks,
                    embedding=embedding_model,
                    persist_directory=persist_directory
                )
            return vectordb
        except Exception as e:
            print(f"Error loading vector db: {e}")
            return Chroma.from_documents(
                documents=chunks,
                embedding=embedding_model,
                persist_directory=persist_directory
            )

