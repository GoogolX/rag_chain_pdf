from langchain_community.llms import CTransformers
import os
import multiprocessing

def load_llm():
    # use cpu cores info for determining how many thread to use
    cpu_count = multiprocessing.cpu_count()
    
    return CTransformers(
        model="TheBloke/Mistral-7B-Instruct-v0.1-GGUF",
        model_file="mistral-7b-instruct-v0.1.Q4_K_M.gguf",
        config={
            # speed related params
            "threads": min(cpu_count - 3, 8),  # use available cores but leave three free (tune based on pref)
            "batch_size": 8,
            "gpu_layers": 0,                   # note: set to 35 if you have GPU
            "stream": False,                   # streaming is sposed to make response faster
            
            # quality related params
            "temperature": 0.4,
            "top_k": 15,                       # these two params can be used to tune response quality (creative vs focused)
            "top_p": 0.9,                      # these two params can be used to tune response quality (creative vs focused)
            "repetition_penalty": 1.1,         # try to prevent repetitive text
            
            # context settings
            "context_length": 1024,
            "max_new_tokens": 256,            
            "reset": False,                    # remember context between runs
        }
    )