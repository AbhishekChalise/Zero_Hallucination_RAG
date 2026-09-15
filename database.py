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


def search_database(query: str, k:int = 5, fetch_k: int = 150):
    query_vector = llm.embedder_model([query])[0] # returns[[0.5]]  
    db = lancedb.connect("rag_data")
    table = db.open_table("rag_corpus")
    results = table.search(query_vector).limit(fetch_k).to_list()
    bm25_results = table.search(query, query_type = "fts" ).limit(fetch_k).to_list()

    clean_results = []
    fused_scores = {}

    for rank, item in enumerate(results):
        score = 1 / (config.rrf_k + rank)

        fused_scores[item["text"]] = {
            "score": score,
            "title": item["title"],
            "summary": item["summary"]
        }

    for rank, item in enumerate(bm25_results):
        score = 1 / (config.rrf_k + rank)

        if item["text"] in fused_scores:
            fused_scores[item["text"]]["score"] += score

        else:
            fused_scores[item["text"]] = {
                "score": score,
                "title": item["title"],
                "summary": item["summary"]
            }

    sorted_fuse = sorted(fused_scores.items(), key = lambda x: x[1]["score"], reverse = True)

    candidate_text = [text for text,data in sorted_fuse]
    ranked_docs = llm.rerank(query, candidate_text)
    top_k = ranked_docs[:k]


    for text, reranker_score in top_k:
        clean_results.append({
            "text": text,
            "title": fused_scores[text]["title"],
            "summary": fused_scores[text]["summary"],
            "score": reranker_score
        })

    return clean_results

if __name__ == "__main__":
    # 1. Make sure the database is already built!
    # build_database()
    
    print("\n=== RETRIEVAL & RERANKING TEST ===")
    question = "Were Scott Derrickson and Ed Wood of the same nationality?"
    
    # We use k=3 to match the blog's top-3 output
    results = search_database(question, k=3)
    
    print(f"Q: {question}")
    print(f"top-3 reranked:")
    
    # Print it beautifully!
    for res in results:
        print(f"   ({res['score']:.3f}) {res['title']}")