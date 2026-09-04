import inspect
import re
import asyncio
from data_loader import Passage
from transformers import AutoTokenizer
from dataclasses import dataclass
from main import llm

max_token = 256
tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2")

@dataclass
class Chunk:
    chunk_id: str
    passage_id: str
    title: str
    text: str
    summary: str


async def generate_summary(chunks: list):
    async def process_chunk(chunk: Chunk):
        prompt = f"Here is a document titled '{chunk.title}':\n{chunk.text}\n\nGive a short, single-sentence context (<=25 words) that situates this chunk within the document so it can be retrieved on its own. Answer with the sentence only."
        summary = await llm.chat(system_chat="Write clear concise retrieval context.", user_chat=prompt, temperature=0)
        chunk.summary = summary
        return chunk
    task = [process_chunk(c) for c in chunks]
    updated_chunks = await asyncio.gather(*task)
    return updated_chunks


def smart_chunk_text(passages: str):

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
                            text=" ".join(current_sentences),
                            summary= ""
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
                    text=" ".join(current_sentences),
                    summary=""
                )
            )

    chunks = asyncio.run(generate_summary(chunks))

    return chunks


if __name__ == "__main__":
    test_passages = [
        Passage(id="1", title="Scott Derrickson", text="Scott Derrickson is a director. He is American. He directed Doctor Strange."),
        Passage(id="2", title="Ed Wood", text="Ed Wood was a filmmaker. He was American. He made Plan 9.")
    ]
    
    final_chunks = smart_chunk_text(test_passages)
    
    for c in final_chunks:
        print(f"\nTitle: {c.title}")
        print(f"Summary: {c.summary}")
        print(f"Text: {c.text}")