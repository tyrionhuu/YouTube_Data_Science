from llama_cpp import Llama

my_model_path = '../models/Meta-Llama-3-8B-Instruct.Q4_0.gguf'

CONTEXT_SIZE = 512


model = Llama(
    model_path=my_model_path,
    n_ctx=CONTEXT_SIZE
)

output = model(
    "Q: This comment 'you cant love your country only when you win' is made to a political news video titled 'CNN-Full "
    "Speech: President Biden’s 2024 State of the Union address', use only one word to classify the political stance "
    "of this comment."
)

print(output)
