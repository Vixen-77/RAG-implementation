"""
Relevance & Hallucination Graders
Prevent hallucinations by:
- validating retrieval quality
- checking if the final answer is grounded in docs
"""

from typing import List, Dict, Any
from Services.llm_service import call_ollama


def _safe_upper(text: str) -> str:
    return text.upper() if isinstance(text, str) else ""


def grade_context_relevance(query: str, context: str) -> Dict[str, Any]:
    """
    Grade whether the retrieved context can answer the query.

    Returns:
        {
            "relevant": bool,
            "reasoning": str,
            "confidence": str  # "HIGH", "MEDIUM", "LOW"
        }
    """
    # VERY strict for car manuals
    prompt = f"""You are a strict quality checker for a car manual diagnostic assistant.

QUESTION:
{query}

CONTEXT (EXCERPT):
{context[:2000]}

TASK:
Decide if the CONTEXT directly addresses the QUESTION.

RULES (be strict):
- Answer "YES" only if the context clearly mentions the same type of fault or situation as the question
  (e.g., smoke under hood, DPF warning light, starting problem, battery flat, etc.).
- Generic maintenance tips, winter precautions, or unrelated systems = NOT RELEVANT.
- If the context talks about different topics (e.g., diesel winter tips, hazard lights, pollution)
  while the question is about a blackout or engine smoke, ANSWER "NO".
- "Might be related" or "indirectly related" = NO.

OUTPUT FORMAT (exactly):
RELEVANT: YES or NO
CONFIDENCE: HIGH, MEDIUM, or LOW
REASONING: One sentence explaining why.
"""

    raw = call_ollama(prompt, model="llama3.2")
    response = _safe_upper(raw)

    relevant = False
    if "RELEVANT:" in response:
        line = response.split("RELEVANT:")[1].split("\n")[0].strip()
        relevant = "YES" in line

    confidence = "LOW"
    if "CONFIDENCE:" in response:
        conf_line = response.split("CONFIDENCE:")[1].split("\n")[0].strip()
        if "HIGH" in conf_line:
            confidence = "HIGH"
        elif "MEDIUM" in conf_line:
            confidence = "MEDIUM"

    reasoning = "No reasoning provided"
    for line in response.split("\n"):
        if "REASONING:" in line:
            reasoning = line.split("REASONING:")[1].strip()
            break

    return {
        "relevant": relevant,
        "confidence": confidence,
        "reasoning": reasoning,
    }


def grade_hallucination(documents: List[Any], generation: str) -> Dict[str, Any]:
    """
    Check if the generation is grounded in the documents.

    Returns:
        {
            "grounded": bool,
            "reasoning": str,
            "confidence": str
        }
    """
    context = "\n\n".join(
        getattr(doc, "page_content", str(doc)) for doc in documents
    )

    prompt = f"""You are a strict fact-checker for a car manual RAG system.

DOCUMENTS:
{context[:4000]}

GENERATION:
{generation}

RULES:
- Answer "YES" if every factual claim in the generation can be found in the DOCUMENTS.
- Answer "NO" if the generation includes information NOT present in the documents (hallucination).
- Answer "NO" if the generation contradicts the documents.
- Ignore grammatical differences or minor paraphrasing.

OUTPUT FORMAT (exactly):
GROUNDED: YES or NO
CONFIDENCE: HIGH, MEDIUM, or LOW
REASONING: One sentence explaining why.
"""

    raw = call_ollama(prompt, model="llama3.2")
    response = _safe_upper(raw)

    grounded = False
    if "GROUNDED:" in response:
        line = response.split("GROUNDED:")[1].split("\n")[0].strip()
        grounded = "YES" in line

    confidence = "LOW"
    if "CONFIDENCE:" in response:
        conf_line = response.split("CONFIDENCE:")[1].split("\n")[0].strip()
        if "HIGH" in conf_line:
            confidence = "HIGH"
        elif "MEDIUM" in conf_line:
            confidence = "MEDIUM"

    reasoning = "No reasoning provided"
    for line in response.split("\n"):
        if "REASONING:" in line:
            reasoning = line.split("REASONING:")[1].strip()
            break

    return {
        "grounded": grounded,
        "confidence": confidence,
        "reasoning": reasoning,
    }


def grade_answer_question(question: str, generation: str) -> Dict[str, Any]:
    """
    Check if the generation addresses the question.

    Returns:
        {
            "useful": bool,
            "reasoning": str,
            "confidence": str
        }
    """
    prompt = f"""You are a grader for a car manual assistant. Check if the GENERATION answers the QUESTION.

QUESTION:
{question}

GENERATION:
{generation}

RULES:
- Answer "YES" if the generation directly addresses the question.
- Answer "NO" if the generation is evasive, off-topic, or pure boilerplate.
- Answer "YES" if the generation correctly states that the information is missing from the context.

OUTPUT FORMAT (exactly):
USEFUL: YES or NO
CONFIDENCE: HIGH, MEDIUM, or LOW
REASONING: One sentence explaining why.
"""

    raw = call_ollama(prompt, model="llama3.2")
    response = _safe_upper(raw)

    useful = False
    if "USEFUL:" in response:
        line = response.split("USEFUL:")[1].split("\n")[0].strip()
        useful = "YES" in line

    confidence = "LOW"
    if "CONFIDENCE:" in response:
        conf_line = response.split("CONFIDENCE:")[1].split("\n")[0].strip()
        if "HIGH" in conf_line:
            confidence = "HIGH"
        elif "MEDIUM" in conf_line:
            confidence = "MEDIUM"

    reasoning = "No reasoning provided"
    for line in response.split("\n"):
        if "REASONING:" in line:
            reasoning = line.split("REASONING:")[1].strip()
            break

    return {
        "useful": useful,
        "confidence": confidence,
        "reasoning": reasoning,
    }
