from llama_cpp import Llama
import os
import pandas as pd
import re

my_model_path = '../models/Meta-Llama-3-8B-Instruct.Q8_0.gguf'

CONTEXT_SIZE = 512

def target_stance_detection(text: str, video_title: str):
    system_prompt = """You are a political stance classifier tasked with labeling comments on US 
political news videos from YouTube channels like CNN, Fox News, MSNBC, etc."""

    example_prompt = """
I have a comment from a YouTube video titled "CNN-Full Speech: President Biden’s 2024 State of the Union address":
Comment: "i think this was the worst state of the union i have ever watched"

Based on the information about the comment above, please label this comment as one of the following: 
conservative, liberal, or other. Note that you only return the label and nothing more."""

    main_prompt = """
Q: {video_title}
Comment: {text}
A: """

    few_shot_examples = """
Q: CNN-Full Speech: President Biden’s 2024 State of the Union address
Comment: i think this was the worst state of the union i have ever watched
A: CONSERVATIVE
###

Q: CNN-Full Speech: President Biden’s 2024 State of the Union address
Comment: this is why trump wanted to build the wall
A: CONSERVATIVE
###

Q: CNN-Full Speech: President Biden’s 2024 State of the Union address
Comment: go joe biden im voting for you
A: LIBERAL
###

Q: CNN-Full Speech: President Biden’s 2024 State of the Union address
Comment: a thief among thieves
A: LIBERAL
###

Q: CNN-Full Speech: President Biden’s 2024 State of the Union address
Comment: we need communism fuck biden and trump
A: OTHER
###"""

    # Combine the prompts
    prompt = system_prompt + example_prompt + few_shot_examples + main_prompt.format(video_title=video_title, text=text)

    # Initialize the Llama model
    model = Llama(
        model_path=my_model_path,
        n_ctx=CONTEXT_SIZE,
        echo=True,
        verbose=True
    )

    try:
        # Use the model to predict the political stance
        response = model(
            prompt,
            temperature=0.5,
            stop=['\n', '#'],
        )["choices"][0]["text"]
    except ValueError as e:
        print(f"Error occurred: {e}")
        print("Skipping...")
        return "OTHER"

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

target_stance_detection('pinocchio show', 'CNN-Full Speech: President Biden’s 2024 State of the Union address')
