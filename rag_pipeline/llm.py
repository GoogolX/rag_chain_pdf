from langchain.llms import CTransformers
from accelerate import Accelerator



def load_llm():
    accelerator = Accelerator()

    config = {'max_new_tokens': 2048, 'repetition_penalty': 1.1, 'context_length': 2048, 'temperature':0.0, 'gpu_layers':50}
    llm = CTransformers(
          model="TheBloke/Mistral-7B-Instruct-v0.1-GGUF",
          model_file="mistral-7b-instruct-v0.1.Q4_K_M.gguf",
          config=config
          )

    llm, config = accelerator.prepare(llm, config)
    return llm

