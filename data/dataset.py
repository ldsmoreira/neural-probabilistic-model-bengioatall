import re

# Build vocabulary function
def build_vocabulary():
    with open("data/raw/ceten.xml", "r") as file:
        text = file.read()

    # Step 1: Clean the text replacing <s> and </s> tags and converting to lower case
    text = re.sub(r"<s>|</s>", "", text).lower()  # Remove <s> tags and convert to lower case

    # Step 2: Tokenize the text by simply splitting it by spaces
    words = text.split()

    # Step 3: Sort the words by alphabetical order
    words.sort()

    # Step 4: Build the vocabulary
    vocabulary = {word: idx for idx, word in enumerate(set(words))}

    # Optional: Add a special token for unknown words
    vocabulary["<unk>"] = len(vocabulary)

    return vocabulary

if __name__ == "__main__":
    vocabulary = build_vocabulary()
    print("Vocabulary 10 first elements:", {k: vocabulary[k] for k in list(vocabulary)[:10]})
