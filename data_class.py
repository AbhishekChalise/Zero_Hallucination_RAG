from dataclasses import dataclass

@dataclass
class Config:
    mode: str = "api"  
    vllm_base_url: str = "http://localhost:8000/v1"
    vllm_gen_model: str = "Qwen/Qwen2.5-7B-Instruct"
    vllm_embedding_model: str = "BAAI/bge-m3"

    gen_model: str = "openai/gpt-oss-20b"
    embedding_model: str = "models/gemini-embedding-2"
    reranker_model: str = "BAAI/bge-reranker-v2-m3"

    chunk_tokens: int = 256 
    chunk_overlap: int = 32
    retrieve_k: int = 150
    rrf_k: int = 60
    rerank_top_k: int = 20

    max_hops: int = 3
    crag_ok: float = 0.7
    crag_bad: float = 0.4

    tau_claim: float = 0.3 
    tau_abstain: float = 0.3
    seed: int = 42

config = Config()


