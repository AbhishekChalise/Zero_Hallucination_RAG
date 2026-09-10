import lancedb, json
from main import llm


def load_dataset():
    with open ('data.json', 'r') as f:
        my_list = json.load(f)

    return my_list



