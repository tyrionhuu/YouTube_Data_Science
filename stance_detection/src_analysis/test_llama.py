from llama_cpp import Llama

my_model_path = '../models/Meta-Llama-3-8B-Instruct.Q8_0.gguf'
CONTEXT_SIZE = 512

model = Llama(model_path=my_model_path, n_ctx=CONTEXT_SIZE)

print(model("The quick brown fox jumps ", stop=["."])["choices"][0].text)