from llama_cpp import Llama

my_model_path = '../models/Meta-Llama-3-70B-Instruct.Q4_0.gguf'
CONTEXT_SIZE = 512

model = Llama(model_path=my_model_path, n_ctx=CONTEXT_SIZE)

output = model(
      "Q: Name the planets in the solar system? A: ", # Prompt
      max_tokens=32, # Generate up to 32 tokens, set to None to generate up to the end of the context window
      stop=["Q:", "\n"], # Stop generating just before the model would generate a new question
      echo=True
) # Generate a completion, can also call create_completion

print(output) # Print the generated completion