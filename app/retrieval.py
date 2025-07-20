import ollama

from load_dataset import get_dataset
from similarity import cosine_similarity
from constants import EMBEDDING_MODEL, LANGUAGE_MODEL, DATASET_PATH_FILE


def get_vector_database():
    VECTOR_DB, dataset = [], get_dataset(DATASET_PATH_FILE)
    for i, chunk in enumerate(dataset):
        embedding = ollama.embed(model=EMBEDDING_MODEL, input=chunk)['embeddings'][0]
        VECTOR_DB.append((chunk, embedding))
        print(f'Added chunk {i+1}/{len(dataset)} to the database')
    return VECTOR_DB


def retrieve(query, top_n=3):
  VECTOR_DB = get_vector_database()
  query_embedding = ollama.embed(model=EMBEDDING_MODEL, input=query)['embeddings'][0]
  # temporary list to store (chunk, similarity) pairs
  similarities = []
  for chunk, embedding in VECTOR_DB:
    similarity = cosine_similarity(query_embedding, embedding)
    similarities.append((chunk, similarity))
  # sort by similarity in descending order, because higher similarity means more relevant chunks
  similarities.sort(key=lambda x: x[1], reverse=True)
  # finally, return the top N most relevant chunks
  return similarities[:top_n]
