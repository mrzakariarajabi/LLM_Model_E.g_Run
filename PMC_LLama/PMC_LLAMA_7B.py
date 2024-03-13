
from dotenv import load_dotenv
from langchain import HuggingFaceHub, LLMChain
from langchain.prompts import PromptTemplate
import transformers
import torch

load_dotenv()

if torch.cuda.is_available():
    device = torch.device("cuda")  # Move model and tensors to GPU
    print("Using GPU for computations")
else:
    device = torch.device("cpu")
    print("GPU not available, using CPU")

tokenizer = transformers.LlamaTokenizer.from_pretrained('chaoyi-wu/PMC_LLAMA_7B')
model = transformers.LlamaForCausalLM.from_pretrained('chaoyi-wu/PMC_LLAMA_7B').to(device)
sentence = 'i got stomach ache ' 
batch = tokenizer(
            sentence,
            return_tensors="pt", 
            add_special_tokens=False
        ).to(device)
with torch.no_grad():
    generated = model.generate(inputs = batch["input_ids"], max_length=200, do_sample=True, top_k=50)
    print('model predict: ',tokenizer.decode(generated[0]))