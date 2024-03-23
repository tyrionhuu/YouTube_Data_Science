import csv
from collections import Counter
import json
import matplotlib.pyplot as plt


def analyze_word_frequency(texts):
    """
    Analyzes the word frequency in a list of texts.

    Args:
        texts: A list of texts to analyze, each text represented as a list of words.

    Returns:
        A list of tuples containing (word, frequency).
    """
    # Flatten the list of lists into a single list of words
    words = [word for text in texts for word in text]

    # Count word frequencies
    word_counts = Counter(words)

    return word_counts.items()


def save_to_csv(data, filename):
    """
    Saves the data to a CSV file.

    Args:
        data: A list of tuples containing the data to save.
        filename: The name of the CSV file.
    """
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Word", "Frequency"])  # Write headers
        writer.writerows(data)


# Load the data from JSON
with open("../preprocessed_data/tokenized_comments.json", "r") as json_file:
    json_data = json.load(json_file)

# Analyze word frequency
word_counts = analyze_word_frequency(json_data)

# Sort word counts by frequency
word_counts = sorted(word_counts, key=lambda x: x[1], reverse=True)

# Save word counts to a CSV file
csv_file = "../preprocessed_data/word_frequency.csv"
save_to_csv(word_counts, csv_file)

print("Word frequency saved to word_frequency.csv")


def load_from_csv(filename):
    """
    Loads data from a CSV file.

    Args:
        filename: The name of the CSV file.

    Returns:
        A dictionary containing word frequencies.
    """
    word_frequencies = {}
    with open(filename, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            word = row['Word']
            frequency = int(row['Frequency'])
            word_frequencies[word] = frequency
    return word_frequencies


def visualize_word_frequency(word_frequencies, top_n=20):
    """
    Visualizes word frequency data using a horizontal bar plot.

    Args:
        word_frequencies: A dictionary containing word frequencies.
        top_n: Number of top words to display.
    """
    # Sort word frequencies by value in descending order
    sorted_word_frequencies = sorted(word_frequencies.items(), key=lambda x: x[1], reverse=True)

    # Get top N words and their frequencies
    top_words = [pair[0] for pair in sorted_word_frequencies[:top_n]]
    top_frequencies = [pair[1] for pair in sorted_word_frequencies[:top_n]]

    # Create horizontal bar plot
    plt.figure(figsize=(10, 6))
    plt.barh(top_words, top_frequencies, color='skyblue')
    plt.xlabel('Frequency')
    plt.ylabel('Word')
    plt.title('Top {} Most Frequent Words'.format(top_n))
    plt.gca().invert_yaxis()  # Invert y-axis to display most frequent words at the top
    plt.savefig('../images/word_frequency.png')
    plt.show()


# Load word frequencies from CSV
word_frequencies = load_from_csv("../preprocessed_data/word_frequency.csv")

# Visualize top N most frequent words using a horizontal bar plot
visualize_word_frequency(word_frequencies, top_n=20)