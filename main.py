from processing import *
from embeddings import Embedder
from cluster import elbow, cluster
import pandas as pd
import numpy as np
import json
import re


def load_embedded_df(csv_path):
	try:
		df = pd.read_csv(csv_path)
		df['embedding'] = df['embedding'].apply(lambda x: np.array(json.loads(x)))
		return df
	except FileNotFoundError:
		return pd.DataFrame()  # Return an empty DataFrame if the file doesn't exist
	
def normalize_condition(sentence):
    if not isinstance(sentence, str):
        print(f"Warning: Non-string input to normalize_condition: {sentence}")
        sentence = str(sentence)  # Convert non-string inputs to string
    sentence = sentence.lower()  # Convert to lowercase
    sentence = re.sub(r'[^a-z\s]', '', sentence)  # Remove non-alphabetic characters
    sentence = re.sub(r'\s+', ' ', sentence).strip()  # Remove extra spaces
    return sentence

def find_overlapping_conditions_with_files(df):
    condition_df = df[df['type'] == 'condition']

    seen_sentences = {}
    overlapping_conditions = {}

    for index, row in condition_df.iterrows():
        original_sentence = row['sentence']
        sentence = normalize_condition(original_sentence)  # Normalize the sentence

        # Ignore conditions with "primary" or "secondary"
        if "primary" in sentence or "secondary" in sentence:
            continue

        file_path = row['file']

        if sentence in seen_sentences:
            if file_path not in seen_sentences[sentence]:
                seen_sentences[sentence].append(file_path)
            overlapping_conditions[sentence] = seen_sentences[sentence]
        else:
            seen_sentences[sentence] = [file_path]

    return overlapping_conditions


def extract_factors_for_condition(embedded_df, input_vector, file_paths):
	"""Extract embeddings for underlying factors associated with a condition across specific files."""
	# Filter for underlying factors from the specified files
	factors = embedded_df[(embedded_df['type'] == "underlying factor") & (embedded_df['file'].isin(file_paths))]
	return factors


if __name__ == "__main__":
    # Load data
    file_path = "../training_20180910"
    df = process_all_files_in_directory(file_path)

    embedder_instance = Embedder()
    embedded_df = load_embedded_df("embedded_sentences.csv")
    if embedded_df.empty:
        embedded_df = embedder_instance.embed_texts_in_df(df, 'sentence')
        embedded_df.to_csv("embedded_sentences.csv", index=False)

    # Example usage:
    overlapping = find_overlapping_conditions_with_files(embedded_df)

    for condition, associated_files in overlapping.items():
        if len(associated_files) > 2:  # Checking for more than 2 associated files
            print(f"\nCondition: {condition}")
            print("Associated files:", ", ".join(associated_files))

            factors_df = extract_factors_for_condition(embedded_df, condition, associated_files)


            # Now use this DataFrame in your clustering functions
            k = elbow(factors_df)
            print(f"Optimal k: {k}")

            condition_embedding = embedder_instance.create_embedding_bert(condition)
            cluster(factors_df, k, condition_embedding)
