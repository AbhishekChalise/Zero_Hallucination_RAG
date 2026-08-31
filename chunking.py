import re
import tiktoken
from dataclasses import dataclass


@dataclass
class Chunk:
    chunk_id: str
    passage_id: str
    title: str
    text: str

