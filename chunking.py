import inspect
import re
from transformers import AutoTokenizer
from dataclasses import dataclass, field
from data_loader import Passage

max_token = 256
tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2")

@dataclass
class Chunk:
    chunk_id: str
    passage_id: str
    title: str
    text: str

def smart_chunk_text(passages: list):

    chunks = []

    for passage in passages:
        passage_id = passage.id if hasattr(passage, 'id') else None
        title = passage.title if hasattr(passage, 'title') else passage.get('title', '')
        text = passage.text if hasattr(passage, 'text') else passage.get('text', '')

        if not text:
            continue

        sentences = re.split(r'(?<=[.!?]) +', text)

        current_sentences = []
        current_token_count = 0
        chunk_id = 0
        count = 0

        for sentence in sentences:
            
            sentence = sentence.strip()
            if not sentence:
                continue

            old_count = count
            count = len(tokenizer.encode(sentence))

            if current_token_count + count <= max_token:
                current_sentences.append(sentence)
                current_token_count += count
            else:
                if current_sentences:
                    chunks.append(
                        Chunk(
                            chunk_id=f"{passage_id}_chunk_{chunk_id}",
                            passage_id=passage_id,
                            title=title,
                            text=" ".join(current_sentences)
                        )
                    )
                    chunk_id += 1
                    current_sentences = [current_sentences[-1]]
                    current_sentences.append(sentence)
                    current_token_count = count + old_count

                else:
                    current_sentences = [sentence]
                    current_token_count = count
            
        if current_sentences:
            
            chunks.append(
                Chunk(
                    chunk_id=f"{passage_id}_chunk_{chunk_id}",
                    passage_id=passage_id,
                    title=title,
                    text=" ".join(current_sentences)
                )
            )


    return chunks

if __name__ == "__main__":
    sample_passages = [
        Passage(
            id="doc_1",
            title="Sample Passage",
            text="This is the first sentence. Here is the second sentence which adds more detail. Finally, this is the third sentence to complete the chunking test."
        )
    ]

    result_chunks = smart_chunk_text(sample_passages)
    for c in result_chunks:
        print(c)