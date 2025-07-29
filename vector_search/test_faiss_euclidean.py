import numpy as np
import faiss
print("Running vector search test euclidean")
# Load the embedding dict
data_dict = np.load("image_embeddings.npy", allow_pickle=True).item()

# Extract filenames and vectors
filenames = list(data_dict.keys())
vectors = np.stack([data_dict[name] for name in filenames]).astype('float32')


# Build FAISS index with Inner Product (works like cosine similarity now)
dim = vectors.shape[1]
index = faiss.IndexFlat(dim)
index.add(vectors)
print(f"Loaded {len(filenames)} normalized vectors into FAISS index.")

# Example query: use first vector (also normalized)
query_vector = vectors[0].reshape(1, -1)

# Search top k most similar
k = 5
distances, indices = index.search(query_vector, k)

# Print results
print("INTERNAL: Top 5 most similar images:")
for i in range(k):
    print(f"{i+1}. Filename: {filenames[indices[0][i]]}, Similarity: {distances[0][i]:.4f}")



check_dict = np.load("image_embeddings_search.npy", allow_pickle=True).item()
print(check_dict)

for key, value in check_dict.items():
    print('Ground Truth:', key)    
    query_vector = value.reshape(1, -1)

    k = 5
    distances, indices = index.search(query_vector, k)

    # Print results
    print("Found only from text: Top 5 most similar images:")
    for i in range(k):
        print(f"{i+1}. Filename: {filenames[indices[0][i]]}, Similarity: {distances[0][i]:.4f}")
