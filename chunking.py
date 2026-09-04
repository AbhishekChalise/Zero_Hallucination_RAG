import re
import os
import json
import asyncio
import inspect
from main import llm
from dataclasses import dataclass
from transformers import AutoTokenizer

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

    checkpoint = []
    processed_passage_ids = set()

    if os.path.exists("data.json"):
        try:
            with open("data.json", "r", encoding="utf-8") as f:
                checkpoint = json.load(f)
                processed_passage_ids = {c["passage_id"] for c in checkpoint}
        except (json.JSONDecodeError, OSError):
            checkpoint = []
            processed_passage_ids = set()

    chunks = []

    for passage in passages:
        passage_id = passage.id if hasattr(passage, 'id') else None
        
        if passage_id in processed_passage_ids:
            continue

        title = passage.title if hasattr(passage, 'title') else passage.get('title', '')
        text = passage.text if hasattr(passage, 'text') else passage.get('text', '')

        if not text:
            continue

        sentences = re.split(r'(?<=[\.!?])(?<!\bDr)(?<!\bMr)(?<!\bMrs) +', text)
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

    if chunks:

        chunks = asyncio.run(generate_summary(chunks))
        new_checkpoint = [{"chunk_id":c.chunk_id, "passage_id":c.passage_id, "title": c.title, "text":c.text, "summary": c.summary} for c in chunks]
        checkpoint.extend(new_checkpoint)

        with open("data.json","w", encoding = "utf-8") as f:
            json.dump(checkpoint, f, indent=4)

        return chunks
    else:
        final_chunk = [Chunk(**c) for c in checkpoint]
        return final_chunk