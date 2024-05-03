from llama_cpp import Llama

my_model_path = '../models/Meta-Llama-3-8B-Instruct.Q4_0.gguf'

CONTEXT_SIZE = 512

def target_stance_detection(text: str, video_title: str):
    system_content = f"""You are master of all knowledge about history, literature, science, social science, philosophy, 
religion, economics, sports, etc."""

    prompt = f"""
# CONTEXT #
We are looking into a huge amount of comments on YouTube videos that are from news channels like CNN, Fox News, MSNBC, 
etc. We are interested in understanding the political stance of the comments. The comments can be about the video,
the news, or the political figures. The political stance can be Conservative, Liberal, or Other.

#################

# OBJECTIVE #
You are a political stance classifier. You are given a political news video. 
Your task is to analyze the given comment and identify its political stance as one of the following: 
Conservative, Liberal, or Other.

#################

# REQUIREMENTS #
You need to provide a one-word answer, either Conservative, Liberal, or Other.

#################
Question: the comment is: {text}, and the video title is: {video_title}, is the comment Conservative, Liberal, or Other?
Answer: """

    # Initialize the Llama model
    model = Llama(
        model_path=my_model_path,
        n_ctx=CONTEXT_SIZE,
        echo=True
    )
    # Use the model to predict the political stance
    response = model(
        system_content + prompt,
        stop=['\n']
    )["choices"][0]["text"]
    stance = response.strip()
    print(stance)
    return stance

target_stance_detection("true american president should not bow down to putin period", "CNN-Full Speech: President Biden’s 2024 State of the Union address")
