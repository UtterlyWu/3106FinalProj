import ProjCodes
from scipy.spatial.distance import cosine
def embeddings():
    # Encode sentences
    model  = ProjCodes.sbert_train()

    while True:
        sentence1 = input("Sentence 1: ")
        sentence2 = input("Sentence 2: ")
        embedding1 = model.encode(sentence1)
        embedding2 = model.encode(sentence2)

        # Compute similarity
        
        similarity = 1 - cosine(embedding1, embedding2)
        print("Agreement" if similarity > 0.8 else "Disagreement")

if __name__ == "__main__":

    embeddings()
    # model  = ProjCodes.sbert_train()

    # embedding1 = model.encode("I don't like that guy")
    # embedding2 = model.encode("I also don't like that guy")

    # # Compute similarity
    # from scipy.spatial.distance import cosine
    # similarity = 1 - cosine(embedding1, embedding2)
    # print("Agreement" if similarity > 0.8 else "Disagreement")