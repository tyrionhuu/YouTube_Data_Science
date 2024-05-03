from llama_cpp import Llama

my_model_path = '../models/codellama-7b.Q5_K_M.gguf'
CONTEXT_SIZE = 512

model = Llama(model_path=my_model_path, n_ctx=CONTEXT_SIZE)

print(model("The quick brown fox jumps ", stop=["."])["choices"][0])