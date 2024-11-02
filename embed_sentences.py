from sentence_transformers import SentenceTransformer
from pymongo import MongoClient

def embed_sentences(collection):
    # 1. Load a pretrained Sentence Transformer model
    model = SentenceTransformer("all-MiniLM-L6-v2", device="cuda")

    # The sentences to encode
    sentences = get_gists(collection)
    # 2. Calculate embeddings by calling model.encode()
    embeddings = model.encode(sentences)
    print(embeddings.shape)

    # 3. Calculate the embedding similarities
    similarities = model.similarity(embeddings, embeddings)
    print(similarities)

def get_gists(collection):
    query = {"has_gist": True,}
    
    posts = list(collection.find(query))
    gists = []
    for post in posts:
        gists.append(post['gist'])
    return gists

if __name__ == "__main__":
    client = MongoClient()
    db = client['reddit']
    collection = db['gist_test']
    embed_sentences(collection)
