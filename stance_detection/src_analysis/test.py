from llama_cpp import Llama
import os
import pandas as pd
import re


my_model_path = '../models/Meta-Llama-3-8B-Instruct.Q4_0.gguf'

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
        verbose=True
    )
    # Use the model to predict the political stance
    response = model(
        system_content + prompt,
        temperature=1.0,
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

comment = 'amazing how much joe and this government cares more about ukraine than our own countrys problems'
video_title = 'CNN-Full Speech: President Biden’s 2024 State of the Union address'
target_stance_detection(comment, video_title)