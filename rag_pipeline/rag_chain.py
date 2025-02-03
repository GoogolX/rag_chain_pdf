from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
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
    
    # Add some debug logs
    print("Getting embedding model...")
    embedding_model = get_embedding_model()
    print("Creating vector database...")
    vector_db = create_vector_db(chunks, embedding_model)
    print("Loading LLM...")
    llm = load_llm()
    print("LLM loaded successfully")

    if prompt is None:
        prompt_template = """Use the following pieces of context to answer the question. If you don't know the answer based on the context provided, just say that you don't know.

Context: {context}
Question: {question}
Answer: """
        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )

    chain_kwargs = {}
    if prompt:
        chain_kwargs["prompt"] = prompt

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vector_db.as_retriever(search_kwargs={"k": k}),
        chain_type=chain_type,
        chain_type_kwargs=chain_kwargs,
        return_source_documents=False
    )
    return qa_chain

# Initialize with Report.pdf
pdf_path = "Report.pdf"
qa_chain = create_qa_chain(pdf_path)

def get_answer(query):
    try:
        # adding some debug logs
        print(f"Processing query: {query}")
        print("Invoking QA chain...")
        result = qa_chain.invoke({"query": query})
        print("Got response from QA chain")
        return result["result"]
    except Exception as e:
        print(f"Error processing query: {str(e)}")
        import traceback
        print(f"Full error: {traceback.format_exc()}")
        return f"Sorry, I encountered an error while processing your question: {str(e)}"
