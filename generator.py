import re
import json
import asyncio
from main import llm
from dataclasses import dataclass

@dataclass
class CitedAnswer:
    text: str
    cited_ids: list
    abstained: bool
    raw: str


ABSTAIN_TOKEN = "INSUFFICIENT_EVIDENCE"
GENERATION_SYSTEM_PROMPT = (
    "You answer strictly from the numbered context passages. Rules:\n"
    "1. Use ONLY facts in the passages, never outside knowledge.\n"
    f"2. If the passages do not contain the answer, reply with exactly: {ABSTAIN_TOKEN}\n"
    "3. Every sentence MUST end with a citation to the passage id(s) it uses, like [abc123def456].\n"
    "4. Be concise and factual."
)

_CITE_RE = re.compile(r"\[([a-zA-Z0-9_ ]+)\]") # Finds the citations inside "[]" this bracket

def format_context(chunks: list) -> str:
    """Formats the retrieved chunks so the LLM knows the IDs."""
    ctx = ""
    for c in chunks:
        # We use the title as the ID because that's what we stored in LanceDB
        ctx += f"[{c['title']}] {c['text']}\n"
    return ctx


# 4. Citation Parser (The Fake Citation Stripper)
def parse_citations(text: str, valid_ids: set) -> tuple:
    """Finds citations, keeps the real ones, strips the fake ones."""
    found = _CITE_RE.findall(text)
    
    # Remove duplicates but keep order
    valid = [c for c in dict.fromkeys(found) if c in valid_ids]
    invalid = [c for c in dict.fromkeys(found) if c not in valid_ids]
    
    cleaned = text
    for bad in invalid:
        cleaned = cleaned.replace(f"[{bad}]", "") # Delete fake citations
        
    return valid, cleaned


class CitedGenerator:
    async def generate(self, question: str, chunks: list) -> CitedAnswer:
        # 1. Format the context with the chunks
        user_prompt = f"Context passages:\n{format_context(chunks)}\n\nQuestion: {question}\n\nAnswer:"
        
        # 2. Call the LLM
        raw = await llm.chat(
            system_chat=GENERATION_SYSTEM_PROMPT, 
            user_chat=user_prompt, 
            temperature=0.0
        )
        raw = raw.strip()
        
        # 3. Check if the LLM chose to abstain
        if ABSTAIN_TOKEN in raw:
            return CitedAnswer(text="", cited_ids=[], abstained=True, raw=raw)
            
        # 4. Parse citations and strip fake ones
        # Our valid IDs are the titles of the chunks we retrieved
        valid_ids = {c["title"] for c in chunks}
        cited, cleaned = parse_citations(raw, valid_ids)
        
        return CitedAnswer(text=cleaned.strip(), cited_ids=cited, abstained=False, raw=raw)


# --- Test it ---
if __name__ == "__main__":
    async def test():
        print("=== CITED GENERATION TEST ===")
        question = "Were Scott Derrickson and Ed Wood of the same nationality?"
        
        # Pretend we retrieved these chunks from LanceDB
        mock_chunks = [
            {"title": "Scott Derrickson", "text": "Scott Derrickson (born July 16, 1966) is an American director.", "summary": "..."},
            {"title": "Ed Wood", "text": "Edward Davis Wood Jr. was an American filmmaker.", "summary": "..."}
        ]
        
        generator = CitedGenerator()
        answer = await generator.generate(question, mock_chunks)
        
        print(f"Abstained: {answer.abstained}")
        print(f"Citations: {answer.cited_ids}")
        print(f"Cleaned Answer: {answer.text}")

    asyncio.run(test())