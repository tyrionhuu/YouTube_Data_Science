import pandas as pd
import re
from llama_cpp import Llama
from sklearn.metrics import accuracy_score

# Model configuration
my_model_path = '../models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf'
CONTEXT_SIZE = 1024
MAX_TOKENS = 1024  # Adjust as needed
TEMPERATURE = 0.5  # Lower temperature for more focused output

# Load the Llama model
llama_model = Llama(model_path=my_model_path, context_size=CONTEXT_SIZE, verbose=True)

def stance_detection(text: str, target: str):
    # Using the recommended prompt format for Llama 3.1
    system_prompt = """<|start_header_id|>system<|end_header_id|>You are a political stance 
    classifier tasked with analyzing comments on political figures based on their content. Classify the stance as 
    'FAVOR', 'AGAINST', or 'NONE' based strictly on the comment's content and context."""

    few_shot_examples = """<|eot_id|><|start_header_id|>user<|end_header_id|>Target: Joe Biden\nText: Joe Biden is 
    looking to gather votes from unsuspecting voters. One must remember, Good Ole Boy Joe supported a Grand Wizard of 
    the KKK. Joe cannot deny it.<|eot_id|><|start_header_id|>assistant<|end_header_id|>AGAINST<|eot_id
    |><|start_header_id|>user<|end_header_id|>Target: Joe Biden\nText: Check out the latest podcast conversation 
    between @JoeBiden and @AndrewYang. #HeresTheDeal #UnitedForJoe #BarnstormersForAmerica 
    #ITrustJoe<|eot_id|><|start_header_id|>assistant<|end_header_id|>FAVOR<|eot_id|><|start_header_id|>user
    <|end_header_id|>Target: Joe Biden\nText: DJT should go to 
    prison.<|eot_id|><|start_header_id|>assistant<|end_header_id|>NONE"""

    user_input = (f"<|eot_id|><|start_header_id|>user<|end_header_id|>Target: Joe Biden\nText: {text}<|eot_id"
                  f"|><|start_header_id|>assistant<|end_header_id|>")

    prompt = f"{system_prompt}{few_shot_examples}{user_input}"

    # Generate prediction
    try:
        result = llama_model(
            prompt=prompt,
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
            echo=False,
            stop=["<"]
        )
        print(result)
        if isinstance(result, dict) and 'choices' in result and len(result['choices']) > 0:
            response = result['choices'][0]['text'].strip()
        else:
            response = str(result).strip()
        print(f"Response: {response}")
    except Exception as e:
        response = f"Error: {e}"
        print(f"Error generating response: {e}")

    # Extract the stance label from the result
    match = re.search(r'\b(FAVOR|AGAINST|NONE)\b', response)
    if match:
        return match.group(1)
    else:
        return "NONE"  # Default to NONE if no match is found

# Load the training data from the specified file path
file_path = '../training_data/raw_train_biden.csv'
test_data = pd.read_csv(file_path)

# Randomly select 50 examples
test_data_sample = test_data.sample(n=200, random_state=42)

# Generate predictions
test_data_sample['predicted_stance'] = test_data_sample.apply(lambda row: stance_detection(row['Tweet'], row['Target']), axis=1)

# Calculate accuracy
accuracy = accuracy_score(test_data_sample['Stance'], test_data_sample['predicted_stance'])
print(f"Accuracy: {accuracy}")

# Display the results
print(test_data_sample)

# Print out the wrongly classified texts
wrong_classifications = test_data_sample[test_data_sample['Stance'] != test_data_sample['predicted_stance']]
print("Wrongly classified texts:")
for index, row in wrong_classifications.iterrows():
    print(f"Tweet: {row['Tweet']}")
    print(f"Target: {row['Target']}")
    print(f"True Stance: {row['Stance']}")
    print(f"Predicted Stance: {row['predicted_stance']}")
    print()
