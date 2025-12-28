import json
from enum import Enum
from .client import call_ollama


class RouteDecision(Enum):
    RAG_NEEDED = "RAG_NEEDED"
    DIRECT_ANSWER = "DIRECT_ANSWER"
    CLARIFICATION_NEEDED = "CLARIFICATION_NEEDED"
    OUT_OF_SCOPE = "OUT_OF_SCOPE"


class QueryRoute:
    RAG_NEEDED = RouteDecision.RAG_NEEDED.value
    DIRECT_ANSWER = RouteDecision.DIRECT_ANSWER.value
    CLARIFICATION_NEEDED = RouteDecision.CLARIFICATION_NEEDED.value
    OUT_OF_SCOPE = RouteDecision.OUT_OF_SCOPE.value


def route_query(query: str, history: list = None) -> dict:
    history_context = _format_history(history) if history else ""

    prompt = f"""You are a routing agent for a Dacia vehicle workshop assistant.
{history_context}
USER QUERY: "{query}"

Choose ONE option:
1. RAG_NEEDED - Needs workshop manual lookup (repairs, specs, troubleshooting)
2. DIRECT_ANSWER - General automotive knowledge (basic concepts)
3. CLARIFICATION_NEEDED - Too vague ("fix it", "help", single words)
4. OUT_OF_SCOPE - Unrelated to vehicles (weather, recipes, etc.)

Respond with JSON only:
{{"decision": "RAG_NEEDED|DIRECT_ANSWER|CLARIFICATION_NEEDED|OUT_OF_SCOPE", "reasoning": "brief", "reformulated_query": "clearer version"}}"""

    try:
        response = _parse_json_response(call_ollama(prompt))
        
        if response.get("decision") not in [r.value for r in RouteDecision]:
            response["decision"] = QueryRoute.RAG_NEEDED
        
        print(f"[ROUTER] '{query[:40]}...' -> {response['decision']}")
        return response
        
    except Exception as e:
        print(f"[ROUTER] Error: {e}, defaulting to RAG")
        return {
            "decision": QueryRoute.RAG_NEEDED,
            "reasoning": "Router error",
            "reformulated_query": query
        }


def generate_direct_answer(query: str, history: list = None) -> str:
    history_text = _format_history(history) if history else ""

    prompt = f"""You are a Dacia vehicle assistant. Answer using general automotive knowledge.
{history_text}
Question: {query}

Keep it concise. If unsure about Dacia specifics, recommend checking the manual."""

    return call_ollama(prompt)


def generate_clarification_request(query: str) -> str:
    prompt = f"""The user asked: "{query}"

This is too vague. Ask for specifics in 1-2 sentences (vehicle model, symptom, goal)."""

    return call_ollama(prompt)


def generate_out_of_scope_response(query: str) -> str:
    return "I specialize in Dacia vehicle repair and maintenance. How can I help with your vehicle?"


def _format_history(history: list, limit: int = 4) -> str:
    if not history:
        return ""
    recent = history[-limit:]
    lines = [f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content'][:100]}" for m in recent]
    return f"\nRecent conversation:\n" + "\n".join(lines) + "\n"


def _parse_json_response(response: str) -> dict:
    response = response.strip()
    if response.startswith("```"):
        response = response.split("```")[1]
        if response.startswith("json"):
            response = response[4:]
    return json.loads(response.strip())
