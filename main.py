import os
import torch 
import aiohttp
from groq import AsyncGroq
from data_class import config
from openai import AsyncOpenAI
from dotenv import load_dotenv
from FlagEmbedding import FlagReranker
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_openai import OpenAIEmbeddings
from vram import vram_snapshot

load_dotenv()

# vram_snapshot("Before Init")

use_fp16 = True if getattr(config, "mode") == "vllm" else False

if getattr(config, "mode") == "vllm":
    client = AsyncOpenAI(base_url = config.vllm_base_url, api_key="EMPTY")
elif getattr(config, "mode") == "api":
    client = AsyncGroq(api_key = os.environ.get("GROQ_API_KEY"))
else:
    client = None

class LocalLLM:

    def __init__(self, model: str, embedding_model: str, reranker_model: str):
        if getattr(config, "mode") == "vllm":
            self.model = config.vllm_gen_model
            # self.embedding_model = config.vllm_embedding_model
            self.embed_tok = AutoTokenizer(embedding_model)
            self.embedding_model = AutoModel.from_pretrained(embedding_model, torch_dtype = torch.float16).to("cuda")

        else:
            self.model = model
            self.embedding_model = embedding_model

        self.reranker = FlagReranker(
            reranker_model,
            use_fp16 = use_fp16
        )

    async def chat(self, system_chat: str, user_chat: str, temperature: float = 0.0):

        response = await client.chat.completions.create(
            model = self.model,
            temperature = temperature,
            messages=[
                {"role": "user", "content": user_chat},
                {"role": "system", "content": system_chat}
            ]
        )
        return response.choices[0].message.content

    def embedder_model(self):
        if getattr(config, "mode") == "vllm":
            return OpenAIEmbeddings(
                model = self.embedding_model,
                openai_api_base=config.vllm_base_url,
                openai_api_key="EMPTY"
            )
        return GoogleGenerativeAIEmbeddings(
            model = self.embedding_model,
        )

    async def rerank(self, query: str, documents: list[str]):
        # vram_snapshot("Before Rerank")
        if getattr(config, "mode") == "vllm":
            tei_url = f"{config.tei_base_url}/rerank"
            payload = {"query": query, "texts": documents}

            async with aiohttp.ClientSession() as session:
                async with session.post(tei_url, json = payload) as response:
                    result = await response.json()
                    scores = [item["score"] for item in result]

        else:
            pairs = [[query, document] for document in documents]

            scores = self.reranker.compute_score(
                pairs,
                normalize = True
            )

        ranked = sorted(
            zip(documents, scores),
            key= lambda x : x[1],
            reverse = True
        )
        # vram_snapshot("After Rerank")

        return ranked

llm = LocalLLM(
    model=config.gen_model,
    embedding_model=config.embedding_model,
    reranker_model=config.reranker_model
)