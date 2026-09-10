import lancedb, json
from main import llm


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





