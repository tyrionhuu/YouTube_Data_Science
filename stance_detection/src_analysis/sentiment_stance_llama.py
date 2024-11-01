from llama_cpp import Llama
import os
import pandas as pd
import re

my_model_path = '../models/Meta-Llama-3-8B-Instruct.Q8_0.gguf'
CONTEXT_SIZE = 512


def stance_detection(text: str, video_title: str):
    system_prompt = """You are a im partial political sentiment classifier tasked with labeling comments on US 
political news videos from YouTube channels like CNN, Fox News, MSNBC, etc."""

    example_prompt = """
I have a comment from a YouTube video , and based on the information about the comment above, please label this comment 
by it's sentiment as one of the following: NEGATIVE, POSITIVE, or NEURAL. 
Note that you only return the label and nothing more."""

    main_prompt = """
Q: {video_title}
Comment: {text}
###
"""

    few_shot_examples = """
Q: CNN-Full Speech: President Biden’s 2024 State of the Union address
Comment: i think this was the worst state of the union i have ever watched
###
NEGATIVE
######

Q: CNN-Full Speech: President Biden’s 2024 State of the Union address
Comment: this is why trump wanted to build the wall
###
NEUTRAL
######

Q: CNN-Full Speech: President Biden’s 2024 State of the Union address
Comment: go joe biden im voting for you
###
POSITIVE
######

Q: CNN-Full Speech: President Biden’s 2024 State of the Union address
Comment: a thief among thieves
###
NEGATIVE
######

Q: CNN-Full Speech: President Biden’s 2024 State of the Union address
Comment: we need communism fuck biden and trump
###
NEGATIVE
######

Q: CNN-Full Speech: President Biden’s 2024 State of the Union address
Comment: oh what a disgrace of a man trump 2024
###
NEGATIVE
######
"""

    # Combine the prompts
    prompt = system_prompt + example_prompt + few_shot_examples + main_prompt.format(video_title=video_title, text=text)

    # Initialize the Llama model
    model = Llama(
        model_path=my_model_path,
        n_ctx=CONTEXT_SIZE,
        echo=True,
        verbose=False
    )

    try:
        # Use the model to predict the political stance
        response = model(
            prompt,
            temperature=0.2,
            stop=['\n', '#'],
        )["choices"][0]["text"]
    except ValueError as e:
        print(f"Error occurred: {e}")
        print("Skipping...")
        return "OTHER"

    # Extract stance from response
    stance = re.search(r'(POSITIVE|NEGATIVE|NEUTRAL)', response.upper())
    if stance:
        stance = stance.group(0)
    else:
        stance = "OTHER"  # Default to "OTHER" if no stance is found

    print(f"{text}")
    print("*******************")
    print(f"{response}")
    print("-------------------")
    print(f"{stance}")
    print("+++++++++++++++++++++")

    return stance


def extract_title(file_name: str):
    # Extract the video title
    title = file_name.rsplit('_', 1)[0]
    return title


# Retrieve comments files and process each one
comments_directory = '../comments/comments2/'
comments_files = [f for f in os.listdir(comments_directory) if f.endswith('.csv')]
output_dir = '../comments/result2/sentiment/'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

for file in comments_files:
    comments = pd.read_csv(comments_directory + file)
    title = extract_title(file)
    # print(title)
    comments['sentiment_llama_8b'] = comments.apply(lambda x: stance_detection(x['comment'], title), axis=1)
    comments.to_csv(output_dir + file, index=False)
