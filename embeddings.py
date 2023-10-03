from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd
from transformers import AutoModel, AutoTokenizer
import logging

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

        embedded_df = pd.DataFrame({
            'original_text': df[text_column],
            'embedding': embeddings
        })

        return embedded_df

    def get_k_most_relevant_texts(self, input_text, embedded_df, k=5):
        input_embedding = self.create_embedding_bert(input_text)
        similarities = []

        for index, row in embedded_df.iterrows():
            similarity = cosine_similarity([input_embedding], [row['embedding']])[0][0]
            similarities.append((row['original_text'], similarity))

        # Sort by similarity and select top k
        similarities.sort(key=lambda x: x[1], reverse=True)
        top_k_texts = similarities[:k]

        return top_k_texts


if __name__ == "__main__":
    your_texts = ["The doctor advised taking vitamins.",
                  "Apples are healthy.",
                  "Hospitals require efficient management.",
                  "Regular check-ups are important for maintaining health.",
                  "Vaccines are crucial for disease prevention."
                  ]
    df = pd.DataFrame(your_texts, columns=['original_text'])

    embedder_instance = Embedder()
    embedded_df = embedder_instance.embed_texts_in_df(df, 'original_text')

    input_text = "How essential are regular health check-ups?"
    k_most_relevant = embedder_instance.get_k_most_relevant_texts(input_text, embedded_df, k=3)

    print("Input text:", input_text)
    print("\nTop k most relevant texts:")
    for i, (text, sim) in enumerate(k_most_relevant):
        print(f"{i+1}. {text} (Similarity: {sim:.4f})")
