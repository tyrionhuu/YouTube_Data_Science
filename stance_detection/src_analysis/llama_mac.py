from llama_cpp import Llama
import os
import pandas as pd
import re

my_model_path = '../models/codellama-7b.Q4_0.gguf'
CONTEXT_SIZE = 512


def target_stance_detection(text: str, video_title: str):
    system_prompt = """You are a political stance classifier tasked with labeling comments on US 
political news videos from YouTube channels like CNN, Fox News, MSNBC, etc."""

    example_prompt = """
I have a comment about US political news:
Comment: "i think this was the worst state of the union i have ever watched"
Based on the information about the comment above, please label this comment's stance towards both Trump and Biden as one of the following: 
- PRO
- ANTI
- OTHER 

Note that you only return the labels in the format: Trump: [label], Biden: [label]"""

    main_prompt = """
Video Title: {video_title}
Comment: {text}
A: """

    few_shot_examples = """
Video Title: CNN-Full Speech: President Biden’s 2024 State of the Union address
Comment: i think this was the worst state of the union i have ever watched
A: Trump: OTHER, Biden: ANTI
###

Video Title: CNN-Full Speech: President Biden’s 2024 State of the Union address
Comment: go joe biden im voting for you
A: Trump: OTHER, Biden: PRO
###

Video Title: CNN-Full Speech: President Biden’s 2024 State of the Union address
Comment: we need communism fuck biden and trump
A: Trump: ANTI, Biden: ANTI
###

Video Title: CNN-Full Speech: President Biden’s 2024 State of the Union address
Comment: trump was a great president, biden is terrible
A: Trump: PRO, Biden: ANTI
###

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
            stop=['#'],
        )["choices"][0]["text"]
    except ValueError as e:
        print(f"Error occurred: {e}")
        print("Skipping...")
        return ("OTHER", "OTHER")

    # Extract stance from response
    trump_stance = re.search(r'Trump: (PRO|ANTI|OTHER)', response)
    biden_stance = re.search(r'Biden: (PRO|ANTI|OTHER)', response)

    if trump_stance and biden_stance:
        trump_label = trump_stance.group(1)
        biden_label = biden_stance.group(1)
        stance = (trump_label, biden_label)
    else:
        stance = ("OTHER", "OTHER")  # Default if no stance is found

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
output_dir = '../comments/result2/stance2/'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# file = 'CNN-Full Speech: President Biden’s 2024 State of the Union address_cleaned.csv'
for file in comments_files:
    comments = pd.read_csv(comments_directory + file)
    title = extract_title(file)

    # Apply the stance detection and split the tuple into two columns
    comments[['stance_trump', 'stance_biden']] = comments.apply(lambda x: pd.Series(target_stance_detection(x['comment'], title)), axis=1)

    # Save the modified DataFrame to a new CSV file
    comments.to_csv(os.path.join(output_dir, file), index=False)
