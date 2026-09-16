import re
import json
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

_CITE_RE = re.compile(r"\[([a-zA-Z0-9]+)\]")

def parse_citations(text: str, valid_ids: set) -> dict:
        found = _CITE_RE.findall(text)

        valid = [c for c in dict.formkeys(found) if c in valid_ids]