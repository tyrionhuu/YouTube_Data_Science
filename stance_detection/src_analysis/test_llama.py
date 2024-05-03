from llama_cpp import Llama

my_model_path = '../models/Meta-Llama-3-70B-Instruct.Q4_0.gguf'
CONTEXT_SIZE = 512

def target_stance_detection(text: str, video_title: str):
    model = Llama(
        model_path=my_model_path,
        n_ctx=CONTEXT_SIZE,
        chat_format='llama-3'
    )
    message = ("You are a political analyst. You are watching a news video titled: " + video_title +
               ". Analyze the following comment and identify the political stance in one word: " + text +
               ". Choose from Conservative, Liberal, Other")
    stance = model(message, stop=["."])["choices"][0]["text"]
    print(text + ": " + stance)
    return stance

target_stance_detection("true american president should not bow down to putin period", "CNN-Full Speech: President Biden’s 2024 State of the Union address")
