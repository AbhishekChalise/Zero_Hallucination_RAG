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

        vectors = llm.embedder_model(text) # returns [[0,1], [0,2]]

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
    table = db.create_table("rag_corpus", data=data_to_insert, mode="overwrite")
    table.create_fts_index("text")
    print("Database and BM25 index built!")

    llm.unload_embedder()

    vram_snapshot("After Freeing Embedder")

def search_database(query: str, k:int = 5):
    query_vector = llm.embedder_model([query])[0] # returns[[0.5]]  
    db = lancedb.connect("rag_data")
    table = db.open_table("rag_corpus")
    results = table.search(query_type = "hybrid")
    final_results = results.vector(query_vector).text(query).limit(5).to_list()

    clean_results = []

    for r in final_results:
        score = r["_relevance_score"]

        clean_item = {
            "text": r["text"],
            "title": r["title"],
            "summary": r["summary"],
            "score": score
        }

        clean_results.append(clean_item)
    return clean_results

if __name__ == "__main__":
    # 1. Build the database (you can comment this out after it runs once!)
    build_database()
    
    # 2. Test the search!
    print("\n=== SEARCH TEST ===")
    question = "Were Scott Derrickson and Ed Wood of the same nationality?"
    results = search_database(question, k=3)
    
    for res in results:
        print(f"Score: {res['score']:.4f} | Title: {res['title']}")
        print(f"Text: {res['text'][:100]}...")
        print("-" * 40)