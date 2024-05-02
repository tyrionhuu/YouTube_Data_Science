from llama_cpp import Llama

my_model_path = '../models/codellama-7b.Q5_K_M.gguf'
CONTEXT_SIZE = 512

llama = Llama(model_path=my_model_path, n_ctx=CONTEXT_SIZE)


def generate_text_from_prompt(
        user_prompt,
        max_tokens=100,
        temperature=0.3,
        top_p=0.1,
        echo=True,
        stop=None):

    if stop is None:
        stop = ["Q", "\n"]
    model_output = llama.create_completion(
        user_prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        echo=echo,
        stop=stop,
    )

    return model_output


if __name__ == "__main__":
    my_prompt = "Q: Name the planets in the solar system? A: "

    print(generate_text_from_prompt(my_prompt))
