import pandas as pd
import re

INPUT_FILE = "dataset/metadata/word_list.txt"
OUTPUT_FILE = "dataset/metadata/words.csv"

words = []

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    for line in f:
        word = line.strip()

        # Skip empty lines
        if not word:
            continue

        # Remove "Back to top"
        if "Back to top" in word:
            continue

        # Skip alphabet headings
        if word in list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"):
            continue

        # Remove numbering like -1, -2, -3
        word = re.sub(r"-\d+$", "", word)

        # Remove extra spaces
        word = " ".join(word.split())

        words.append(word)

# Create DataFrame
df = pd.DataFrame(words, columns=["word"])

# Remove duplicates
df = df.drop_duplicates()

# Sort alphabetically
df = df.sort_values("word")

# Reset index
df = df.reset_index(drop=True)

# Save CSV
df.to_csv(OUTPUT_FILE, index=False)

print("=" * 40)
print("Words before cleaning :", len(words))
print("Unique words          :", len(df))
print("CSV saved successfully!")
print("=" * 40)