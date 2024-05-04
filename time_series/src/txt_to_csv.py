import csv
import os


data_dir = '../headlines/'
output_dir = '../headlines_csv/'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Read the text data from the file
for file in os.listdir(data_dir):
    with open(data_dir + file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    df = []
    # Process the text data and split it into tuples
    for line in lines:
        # Split the line into a tuple
        channel = line.split('-', 1)[0].strip()
        part = line.split('-', 1)[1].strip()
        target = part.split(':', 1)[0].strip()
        headline = part.split(':', 1)[1].strip()

        # Append the tuple to the list
        df.append((channel, target, headline))

    # Write the tuples to a CSV file
    with open(output_dir + file.split('.')[0] + '.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['channel', 'target', 'headline'])
        writer.writerows(df)


