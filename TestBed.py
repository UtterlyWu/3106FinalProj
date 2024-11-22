# Encode sentences
embedding1 = model.encode("Sentence1")
embedding2 = model.encode("Sentence2")

# Compute similarity
from scipy.spatial.distance import cosine
similarity = 1 - cosine(embedding1, embedding2)
print("Agreement" if similarity > 0.8 else "Disagreement")