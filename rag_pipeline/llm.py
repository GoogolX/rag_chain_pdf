from langchain.llms import CTransformers

def load_llm():
    return CTransformers(
        model="TheBloke/Mistral-7B-Instruct-v0.1-GGUF",
        model_file="mistral-7b-instruct-v0.1.Q4_K_M.gguf",
        config={"max_new_tokens": 256, "temperature": 0.7}
    )

