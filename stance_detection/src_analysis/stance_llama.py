import os

import pandas as pd
import re
from llama_cpp import Llama

# Model configuration
my_model_path = '../models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf'
CONTEXT_SIZE = 1024
MAX_TOKENS = 1024  # Adjust as needed
TEMPERATURE = 0.5  # Lower temperature for more focused output

# Load the Llama model
llama_model = Llama(model_path=my_model_path, context_size=CONTEXT_SIZE, verbose=True)


def stance_detection(text: str, video_title: str):
    system_prompt = (
        f"You are a political stance classifier tasked with analyzing comments on political figures. "
        f"The comments are responses to a video titled which will be given to you. Determine the stance for both 'Joe "
        f"Biden' and 'Donald Trump' as 'FAVOR', 'AGAINST', or 'NONE' based on the content of the comment. If the "
        f"comment is ambiguous, try to infer the likely stance based on context and common references."
    )

    few_shot_examples = """<|eot_id|><|start_header_id|>user<|end_header_id|>Text: Joe Biden is looking to gather
    votes from unsuspecting voters. One must remember, Good Ole Boy Joe supported a Grand Wizard of the KKK. Joe
    cannot deny it.<|eot_id|><|start_header_id|>assistant<|end_header_id|>Biden: AGAINST; Trump:
    NONE<|eot_id|><|start_header_id|>user<|end_header_id|>Text: Check out the latest podcast conversation between
    @JoeBiden and @AndrewYang. #HeresTheDeal #UnitedForJoe #BarnstormersForAmerica
    #ITrustJoe<|eot_id|><|start_header_id|>assistant<|end_header_id|>Biden: FAVOR; Trump:
    NONE<|eot_id|><|start_header_id|>user<|end_header_id|>Text: DJT should go to
    prison.<|eot_id|><|start_header_id|>assistant<|end_header_id|>Biden: NONE; Trump:
    AGAINST<|eot_id|><|start_header_id|>user<|end_header_id|>Text: Make America great again! Trump
    2020!<|eot_id|><|start_header_id|>assistant<|end_header_id|>Biden: NONE; Trump: FAVOR"""

    user_input = f"""<|eot_id|><|start_header_id|>user<|end_header_id|>Text: {text}<|eot_id|><|start_header_id
    |>assistant<|end_header_id|>"""

    prompt = f"{system_prompt}{few_shot_examples}\n{user_input}"

    # Generate prediction
    try:
        result = llama_model(
            prompt=prompt,
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
            echo=False,
            stop=["<"]
        )
        if isinstance(result, dict) and 'choices' in result and len(result['choices']) > 0:
            response = result['choices'][0]['text'].strip()
        else:
            response = str(result).strip()
        print(f"Response: {response}")
    except Exception as e:
        response = f"Error: {e}"
        print(f"Error generating response: {e}")

    # Extract the stance labels from the result
    biden_match = re.search(r'Biden: (FAVOR|AGAINST|NONE)', response)
    trump_match = re.search(r'Trump: (FAVOR|AGAINST|NONE)', response)
    biden_stance = biden_match.group(1) if biden_match else "NONE"
    trump_stance = trump_match.group(1) if trump_match else "NONE"

    return biden_stance, trump_stance


def process_csv(input_csv_path, output_csv_path):
    # Extract the video title from the file name
    video_title = os.path.splitext(os.path.basename(input_csv_path))[0]

    # Read the input CSV file
    df = pd.read_csv(input_csv_path)

    # Ensure the CSV contains the required columns
    if 'comment' not in df.columns:
        print("Error: The CSV file must contain a 'comment' column.")
        return

    # Initialize lists to store results
    biden_stances = []
    trump_stances = []

    # Process each comment in the CSV
    for _, row in df.iterrows():
        comment = row['comment']
        biden_stance, trump_stance = stance_detection(comment, video_title)
        biden_stances.append(biden_stance)
        trump_stances.append(trump_stance)

        # Print the outcome of each iteration
        print(f"Comment: {comment}")
        print(f"Stance towards Joe Biden: {biden_stance}")
        print(f"Stance towards Donald Trump: {trump_stance}")
        print()

    # Add the results to the DataFrame
    df['biden_stance'] = biden_stances
    df['trump_stance'] = trump_stances

    # Save the DataFrame to a new CSV file
    df.to_csv(output_csv_path, index=False)
    print(f"Results saved to {output_csv_path}")


def main():
    stance_detection("hes such a clowm", "Fox News-Watch: President Biden's State of the Union address and the GOP's response_classified")
    # comments_dir = "../comments/state_of_union/"
    # files = [f for f in os.listdir(comments_dir) if f.endswith('cleaned.csv')]
    # for file in files:
    #     input_csv = comments_dir + file
    #     output_csv = comments_dir + file.replace('cleaned', 'classified')
    #     process_csv(input_csv, output_csv)
    # Paths to the input and output CSV files
    # input_csv = ("../comments/trump_guilty/CNN-Donald Trump convicted of falsifying business records in hush money "
    #              "scheme_cleaned.csv")
    # output_csv = ('../comments/trump_guilty/CNN-Donald Trump convicted of falsifying business records in hush money '
    #               'scheme_cleaned_classified_comments.csv')  # Replace with your desired output CSV file path
    #
    # Process the CSV file and classify comments
    # process_csv(input_csv, output_csv)


if __name__ == '__main__':
    main()
