import json
import csv
from langdetect import detect
import nltk
import re
import os

output_directory = '../comments/state_of_union/'
data_directory = '../original_data/state_of_union/'
if not os.path.exists(output_directory):
    os.makedirs(output_directory)
# Load the json files from the data directory
files = os.listdir(data_directory)
json_files = [f for f in files if f.endswith('.json')]


def clean_text(text):
    # Remove emojis, special symbols, and extra whitespaces
    cleaned_text = re.sub(r'[^\w\s]', '', text)
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text)
    # remove punctuations
    cleaned_text = re.sub(r'[^\w\s]', '', cleaned_text)
    # Remove leading and trailing whitespaces
    cleaned_text = cleaned_text.strip()
    # Convert to lowercase
    cleaned_text = cleaned_text.lower()
    return cleaned_text


def tokenize_text(text):
    # Tokenize the text
    tokens = nltk.word_tokenize(text)
    return tokens


def remove_stopwords(tokens):
    # Remove stopwords
    stop_words = set(nltk.corpus.stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word not in stop_words]
    return filtered_tokens


def lemmatize_tokens(tokens):
    # Lemmatize the tokens
    lemmatizer = nltk.stem.WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return lemmatized_tokens


for json_file in json_files:
    with open(data_directory + json_file, 'r') as f:
        data = json.load(f)
    comments = data['video_comments']
    cleaned_comments = []
    for comment in comments:
        comment_text = comment['comment_text']
        try:
            lang = detect(comment_text)
            if lang == 'en':
                cleaned_comment = clean_text(comment_text)
                cleaned_comments.append(cleaned_comment)
        except:
            # If language detection fails, skip the comment
            pass

    # Save cleaned comments to a new CSV file, add a header
    with open(output_directory + "{}_cleaned.csv".format(json_file.split('.')[0]), 'w', newline='') as cleaned_csv_file:
        writer = csv.writer(cleaned_csv_file)
        writer.writerow(['comment'])
        for comment in cleaned_comments:
            writer.writerow([comment])

    tokenized_comments = []
    for comment in cleaned_comments:
        tokens = tokenize_text(comment)
        tokens = remove_stopwords(tokens)
        tokens = lemmatize_tokens(tokens)
        tokenized_comments.append(tokens)

    # Save tokenized comments to a new CSV file
    with open(output_directory + "{}_tokenized.csv".format(json_file.split('.')[0]), 'w',
              newline='') as tokenized_csv_file:
        writer = csv.writer(tokenized_csv_file)
        for tokens in tokenized_comments:
            writer.writerow(tokens)
