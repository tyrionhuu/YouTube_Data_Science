from llama_cpp import Llama
import os
import pandas as pd
import re

my_model_path = '../models/Meta-Llama-3-8B-Instruct.Q8_0.gguf'
CONTEXT_SIZE = 512


def target_stance_detection(text: str, video_title: str):
    system_prompt = """You are a political stance classifier tasked with labeling comments on US 
political news videos from YouTube channels like CNN, Fox News, MSNBC, etc."""

    detailed_instructions = """
Your task is to analyze the sentiment of YouTube comments towards two political figures, Trump and Biden, based on the content of the comments and the context provided by the video title.
You will be given a comment and the title of the video it was posted under.
For each comment, you need to determine the stance towards Trump and Biden, and categorize each stance as one of the following:
- PRO: The comment expresses positive sentiment or support.
- ANTI: The comment expresses negative sentiment or opposition.
- OTHER: The comment is neutral, ambiguous, or unrelated.

Consider the video title carefully as it provides important context that may influence the stance expressed in the comment.

You should format your response strictly as: Trump: [label], Biden: [label]

Examples are provided below for guidance."""

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

Video Title: Fox News - Breaking: Trump announces 2024 run for president
Comment: finally someone who can save this country
A: Trump: PRO, Biden: OTHER
###

Video Title: Fox News - Breaking: Trump announces 2024 run for president
Comment: this is a disaster, we need new leadership
A: Trump: ANTI, Biden: OTHER
###

"""

    main_prompt = """
Video Title: {video_title}
Comment: {text}
A: """

    # Combine the prompts
    prompt = system_prompt + detailed_instructions + few_shot_examples + main_prompt.format(video_title=video_title, text=text)

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
            temperature=0.2,
            stop=['#', '\n\n'],
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
        stance = ("NONE", "NONE")  # Default if no stance is found

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

for file in comments_files:
    comments = pd.read_csv(comments_directory + file)
    title = extract_title(file)

    # Apply the stance detection and split the tuple into two columns
    comments[['stance_trump', 'stance_biden']] = comments.apply(lambda x: pd.Series(target_stance_detection(x['comment'], title)), axis=1)

    # Save the modified DataFrame to a new CSV file
    comments.to_csv(os.path.join(output_dir, file), index=False)
