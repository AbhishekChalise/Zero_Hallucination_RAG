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

_CITE_RE = re.compile(r"\[([a-zA-Z0-9_ ]+)\]") 

def format_context(chunks: list) -> str:
    ctx = ""
    for c in chunks:
        ctx += f"[{c['title']}] {c['text']}\n"
    return ctx


def parse_citations(text: str, valid_ids: set) -> tuple:
    """Finds citations, keeps the real ones, strips the fake ones."""
    found = _CITE_RE.findall(text)
    valid = [c for c in dict.fromkeys(found) if c in valid_ids]
    invalid = [c for c in dict.fromkeys(found) if c not in valid_ids]
    
    cleaned = text
    for bad in invalid:
        cleaned = cleaned.replace(f"[{bad}]", "") 
        
    return valid, cleaned


class CitedGenerator:
    async def generate(self, question: str, chunks: list) -> CitedAnswer:
        user_prompt = f"Context passages:\n{format_context(chunks)}\n\nQuestion: {question}\n\nAnswer:"
        
        raw = await llm.chat(
            system_chat=GENERATION_SYSTEM_PROMPT, 
            user_chat=user_prompt, 
            temperature=0.0
        )
        raw = raw.strip()
        
        if ABSTAIN_TOKEN in raw:
            return CitedAnswer(text="", cited_ids=[], abstained=True, raw=raw)
            
        valid_ids = {c["title"] for c in chunks}
        cited, cleaned = parse_citations(raw, valid_ids)
        
        return CitedAnswer(text=cleaned.strip(), cited_ids=cited, abstained=False, raw=
                           raw)

