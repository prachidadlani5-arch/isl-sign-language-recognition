import pandas as pd
import re

# -----------------------------
# Load CSV files
# -----------------------------
words = pd.read_csv("dataset/metadata/words.csv")
videos = pd.read_csv("dataset/metadata/video_links.csv")

# -----------------------------
# Cleaning function
# -----------------------------
def clean(text):
    text = str(text).lower()

    # Remove -1, -2, etc.
    text = re.sub(r"-\d+", "", text)

    # Remove anything inside brackets
    text = re.sub(r"\(.*?\)", "", text)

    # Keep only letters, numbers and spaces
    text = re.sub(r"[^a-z0-9 ]", " ", text)

    # Remove extra spaces
    text = " ".join(text.split())

    return text


# -----------------------------
# Clean words and titles
# -----------------------------
words["clean"] = words["word"].apply(clean)
videos["clean"] = videos["title"].apply(clean)

# -----------------------------
# Remove duplicate words/videos
# -----------------------------
words = words.drop_duplicates(subset="clean")
videos = videos.drop_duplicates(subset="clean")

# -----------------------------
# Match words with videos
# -----------------------------
matched = pd.merge(
    words,
    videos,
    on="clean",
    how="inner"
)

# Remove duplicate matches
matched = matched.drop_duplicates(subset="clean")

# -----------------------------
# Find unmatched words
# -----------------------------
matched_words = set(matched["clean"])

unmatched = words[
    ~words["clean"].isin(matched_words)
][["word"]]

# -----------------------------
# Save CSV files
# -----------------------------
matched[["word", "title", "video_id", "url"]].to_csv(
    "dataset/metadata/matched_videos.csv",
    index=False
)

unmatched.to_csv(
    "dataset/metadata/unmatched_words.csv",
    index=False
)

# -----------------------------
# Print statistics
# -----------------------------
print("=" * 50)
print("Total words :", len(words))
print("Matched     :", len(matched))
print("Unmatched   :", len(unmatched))
print("Accuracy    : {:.2f}%".format((len(matched) / len(words)) * 100))
print("=" * 50)