from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd
from transformers import AutoModel, AutoTokenizer
import logging
import json

logging.getLogger('transformers.modeling_utils').setLevel(logging.ERROR)


class Embedder:
    def __init__(self):
        model_name = "emilyalsentzer/Bio_ClinicalBERT"
        self.model = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def create_embedding_bert(self, text):
        inputs = self.tokenizer(text, return_tensors="pt")
        outputs = self.model(**inputs)
        embedding = outputs.last_hidden_state.mean(dim=1).detach().numpy()
        embedding = embedding.flatten()
        return embedding

    def embed_texts_in_df(self, df, text_column):
        embeddings = []
        for text in df[text_column]:
            embedding = self.create_embedding_bert(text)
            embeddings.append(embedding)

        df['embedding'] = [json.dumps(embed.tolist()) for embed in embeddings]
        return df
    
    def get_k_most_similar_factors_for_condition_vector(self, embedded_df, input_vector, file_paths, k=5):
        # Filter for underlying factors from the specified files
        factors = embedded_df[(embedded_df['type'] == "underlying factor") & (embedded_df['file'].isin(file_paths))]

        results = {}  # To store the results
        similarities = []

        # Reshape the input vector for compatibility with cosine_similarity
        input_vector_reshaped = input_vector.reshape(1, -1)

        for _, factor_row in factors.iterrows():
            factor_text = factor_row['sentence']
            factor_embedding = np.array(factor_row['embedding']).reshape(1, -1)

            similarity = cosine_similarity(input_vector_reshaped, factor_embedding)
            similarities.append((factor_text, similarity))

        # Sort by similarity and select top k
        similarities.sort(key=lambda x: x[1], reverse=True)
        top_k_factors = similarities[:k]

        # Return the results (could be associated with a condition text or an ID for reference if needed)
        return top_k_factors

