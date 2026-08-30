from groq import Groq 
from data_class import config
from dotenv import load_dotenv
from FlagEmbedding import FlagReranker

from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()

client = Groq()

class LocalLLM:

    def __init__(self, model: str, embedding_model: str, reranker_model: str):
        self.model = model
        self.embedding_model = embedding_model
        self.reranker = FlagReranker(
            reranker_model,
            use_fp16 = False
        )

    def chat(self, system_chat: str, user_chat: str, temperature: float = None):

        response = client.chat.completions.create(
            model = self.model,
            temperature = temperature,
            messages=[
                {"role": "user", "content": user_chat},
                {"role": "system", "content": system_chat}
            ]
        )
        return response.choices[0].message.content

    def embedder_model(self):
        return GoogleGenerativeAIEmbeddings(
            model = self.embedding_model,
        )

    def rerank(self, query: str, documents: list[str]):

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

        return ranked

llm = LocalLLM(
    model=config.gen_model,
    embedding_model=config.embedding_model,
    reranker_model=config.reranker_model
)

if __name__ == "__main__":

    try:
        result = llm.chat(
            system_chat="Reply only with: 'True'",
            user_chat="Hello"
        )
        print(f"[Groq] up = True")
        print(f"Response: {result}")

    except Exception as e:
        print(f"[Groq] up = False")
        print(f"Error: {e}")


    # Test Gemini Embedding
    try:
        embedder = llm.embedder_model()

        vector = embedder.embed_query("Hello")

        print(f"[Gemini] up = True")
        print(f"Vector dimensions: {len(vector)}")

    except Exception as e:
        print(f"[Gemini] up = False")
        print(f"Error: {e}")


    # Test Reranker
    try:
        documents = [
            "JWT is used for authentication.",
            "Python is a programming language.",
            "JWT contains a signature and claims."
        ]

        ranked = llm.rerank(
            query="What is JWT authentication?",
            documents=documents
        )

        print(f"[Reranker] up = True")

        for document, score in ranked:
            print(f"{score:.4f} -> {document}")

    except Exception as e:
        print(f"[Reranker] up = False")
        print(f"Error: {e}")