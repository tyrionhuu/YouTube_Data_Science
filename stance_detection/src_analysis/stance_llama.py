from llama_cpp import Llama
import os
import pandas as pd
import re


my_model_path = '../models/Meta-Llama-3-8B-Instruct.Q8_0.gguf'

CONTEXT_SIZE = 512

def target_stance_detection(text: str, video_title: str):
    system_content = f"""
You are master of all knowledge about US politics and history, literature, science, social science, philosophy,"""

    prompt = f"""
# CONTEXT #
We are looking into a huge amount of comments on YouTube videos that are from US news channels like CNN, Fox News, 
MSNBC, etc. We are interested in understanding the political stance of the comments. The comments can be about the 
video,the news, or the political figures. The political stance can be conservative, liberal, or other

#################

# OBJECTIVE #
You are a political stance classifier. You are given a US political news video. 
Your task is to analyze the given comment and identify its political stance in the US political spectrum 
as one of the following: conservative, liberal, or other

#################

# REQUIREMENTS #
You need to provide a one-word answer, either conservative, liberal, or other

#################
Question: the comment is: {text}, and the video title is: {video_title}, is the comment conservative, liberal, or other?
Answer: The comment is """

    # Initialize the Llama model
    model = Llama(
        model_path=my_model_path,
        n_ctx=CONTEXT_SIZE,
        echo=True,
        verbose=False
    )
    if len(system_content + prompt) > 4096:
        print("Text too long, skipping")
        return "OTHER"
    # Use the model to predict the political stance
    response = model(
        system_content + prompt,
        temperature=0.5,
        stop=['.'],
    )["choices"][0]["text"]

    # Extract stance from response
    stance = re.search(r'(CONSERVATIVE|LIBERAL|OTHER)', response.upper())
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

# target_stance_detection("true american president should not bow down to putin period", "CNN-Full Speech: President
# Biden’s 2024 State of the Union address")
comments_directory = '../preprocessed_comments/'
comments_files = [f for f in os.listdir(comments_directory) if f.endswith('liked.csv')]
original_directory = '../data2/'
original_files = [f for f in os.listdir(original_directory) if f.endswith('.json')]
titles = [f.split('.')[0] for f in original_files]

for title in titles:
    comments_file = [f for f in comments_files if title in f][0]
    comments = pd.read_csv(comments_directory + comments_file)
    comments['stance_llama_8b'] = comments.apply(lambda x: target_stance_detection(x['comment'], title), axis=1)
    comments.to_csv(comments_directory + comments_file, index=False)

