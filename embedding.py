from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

print("Embedding model loaded successfully.")

sentences = [
    "I love watching movies.",
    "I enjoy watching films.",
    "I am interested in learning new technologies.",
    "Artificial intelligence is fascinating.",
    "I like building software applications."
]

embeddings = model.encode(sentences)


for i, sentence in enumerate(sentences):

    print("\nSentence:", sentence)

    print("Numerical Embedding:")
    print(embeddings[i])

    print("Number of dimensions:", len(embeddings[i]))


print("\nText has been successfully converted into numerical vectors.")