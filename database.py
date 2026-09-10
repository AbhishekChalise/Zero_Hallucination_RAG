import lancedb, json, torch
from data_class import config
from main import llm
from vram import vram_snapshot

def load_dataset():
    with open ('data.json', 'r') as f:
        my_list = json.load(f)

    return my_list

def hybrid_data_dictonary():

    data = load_dataset()

    hybrid_data = []
    batch_size = 64

    for item in range(0, len(data), batch_size):
        data_i = data[item: item + batch_size]

        text = [chunk["text"] for chunk in data_i]

        vectors = llm.embedder_model(text)

        for chunk, vector in zip(data_i, vectors):
            hybrid_data.append(
                {
                  "vector": vector,
                  "text": chunk["text"],
                  "title": chunk['title'],
                  "summary": chunk['summary']
                }
            )

    return hybrid_data

def build_database():
    vram_snapshot("Before Indexing")

    data_to_insert = hybrid_data_dictonary()

    vram_snapshot("After Indexing")

    db = lancedb.connect("rag_data")
    table = db.create_table("rag_corpus", data=data_to_insert)
    table.create_fts_index("text")
    print("Database and BM25 index built!")

    if getattr(config, "mode") == "vllm":
        del llm.embedding_model          # Destroys the heavy model
        llm.embedding_model = None      # Puts the empty placeholder back!
        torch.cuda.empty_cache()

    vram_snapshot("After Freeing Embedder")