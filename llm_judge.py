import re

JUDGE_PROMPT = (
    "You are a strict fact-checker. Decide whether the CONTEXT supports the CLAIM.\n\n"
    "CONTEXT:\n{context}\n\nCLAIM: {claim}\n\n"
    "Output ONLY a number: 1.0 if the context clearly states or entails the claim, "
    "0.0 if it contradicts or does not mention it, or a value in between."
)

class JudgeVerifier:

    def __init__(self, llm_instance):
        self.llm = llm_instance

    async def score_claim(self, claim: str, context: str) -> float:
        out = await self.llm.chat(
            system_chat="You are a strict faithfulness grader.",
            user_chat=JUDGE_PROMPT.format(context=context[:6000], claim=claim),
            temperature=0.0
        )
        
        match = re.search(r"[01](?:\.\d+)?", out)
        
        if match:
            score = float(match.group())
            return min(1.0, score) 
        return 0.0

    