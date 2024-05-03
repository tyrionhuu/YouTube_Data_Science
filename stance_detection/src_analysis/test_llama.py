from llama_cpp import Llama

my_model_path = '../models/Meta-Llama-3-70B-Instruct.Q4_0.gguf'
CONTEXT_SIZE = 512

def target_stance_detection(text: str, video_title: str):
    model = Llama(
        model_path=my_model_path,
        n_ctx=CONTEXT_SIZE
    )
    # message = "The following text is a comment on the political news video titled: " + video_title + "\nThe
    # description of this video is:" + description + "\nIdentify the stance of the following comment towards this
    # subject: " + text + "\nChoose from Conservative, Liberal, Other"
    message = ("This news video titled " + video_title + (". Analyze the following comment and identify the political "
                                                         "stance (liberal, conservative, etc.) in one word: ") + text
               + "\nChoose from Conservative, Liberal, Other")
    stance = model(message, stop=["."])["choices"][0]["text"]
    print(text + ": " + stance)
    return stance

target_stance_detection("true american president should not bow down to putin period", "CNN-Full Speech: President Biden’s 2024 State of the Union address")
