import json
from main import llm

# There will be two functions Query Router, The Decomposer, False Premise setter
# Query Router ->  ['no_retrieval', 'single_hop', 'multi_hop']
# Decomposer -> If it is multi_hop the LLM Decomposes the query into multiple questions
# False_Premise_Setter - > Does this question assume something that might not be true?" 

ROUTER_PROMPT = """Your task is to analyze the user's question and categorize it into exactly one of the following three routing labels based on its retrieval needs.

Labels:
- no_retrieval: Use for casual greetings, subjective opinions, logical deductions, or general knowledge questions that do not require searching an external document corpus.
- single_hop: Use when the question can be fully answered by retrieving a single specific fact, definition, or localized piece of information from one document.
- multi_hop: Use when the question requires synthesizing information across multiple different documents, comparing distinct entities, or following a multi-step chain of reasoning.

Strict Formatting Constraint: You must output ONLY the exact label string (no_retrieval, single_hop, or multi_hop). Do not include any explanations, quotes, or preamble.

Question: {q}
Label:"""


DECOMPOSE_PROMPT = """ Your task is to analyze a complex, multi-part user question and break it down into a clear sequence of simpler, independent sub-questions that a search engine can easily retrieve.

Guidelines:
- Deconstruct the question into 2 to 4 simple, single-hop queries.
- Ensure each sub-question focuses on retrieving exactly one specific entity, definition, or fact.
- Resolve any pronouns in the sub-questions so they can be searched independently (e.g., replace "his" with the person's name).
- Do not attempt to answer the questions yourself.

Strict Formatting Constraint: Output ONLY a valid JSON list of strings containing the sub-questions. Do not include markdown formatting, preamble, explanations, or the word "json".

Original Question: {q}
Sub-questions:"""


FALSE_PREMISE_PROMPT = """Your task is to analyze the user's question and determine if it is built upon a false premise, invalid assumption, or historically/scientifically impossible fact.

Categories:
- valid_premise: The question makes logical sense and does not assume incorrect facts, even if the answer is complex or unknown.
- false_premise: The question assumes a foundational fact that is definitively false (e.g., "When did Abraham Lincoln invent the airplane?").

Strict Formatting Constraint: You must output ONLY a valid JSON object with exactly two keys: "status" (string: either "valid_premise" or "false_premise") and "correction" (string: a concise 1-sentence correction if false, or null if valid). Do not include markdown formatting, preamble, or explanations outside the JSON.

Question: {q}
Result:"""

class QueryRouter:

    async def route(self, query: str):
        # 1. Format the prompt with the user's question
        user_prompt = ROUTER_PROMPT.format(q=query)
        
        # 2. Call your LLM!
        out = await llm.chat(
            system_chat="You are a precise query classifier.", 
            user_chat=user_prompt, 
            temperature=0.0
        )
        
        # 3. Clean up the output and check if it matches our labels
        out = out.strip().lower()
        for label in ["no_retrieval", "single_hop", "multi_hop"]:
            if label in out:
                return label
                
        # 4. Safe fallback
        return "single_hop"

    async def decompose_query(self, query: str):
        user_prompt = DECOMPOSE_PROMPT.format(q=query)
        
        out = await llm.chat(
            system_chat="You are an expert query planning system.", 
            user_chat=user_prompt, 
            temperature=0.0
        )
        
        try:
            clean_out = out.strip().replace("```json", "").replace("```", "")
            sub_questions = json.loads(clean_out)
            return sub_questions
        except Exception:
            return [query]

    async def detect_false_premise(self, query: str) -> bool:
        user_prompt = FALSE_PREMISE_PROMPT.format(q=query)
        
        out = await llm.chat(
            system_chat="You are an expert logical validation system.", 
            user_chat=user_prompt, 
            temperature=0.0
        )
        
        return "false_premise" in out.lower()