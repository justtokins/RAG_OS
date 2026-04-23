import re
from langchain_community.vectorstores import Chroma
from tools import (
    tool_retrieve_textbook,
    tool_define_term,
    tool_search_wikipedia,
    tool_generate_quiz,
    tool_compare_concepts,
)

DEFINITION_TRIGGERS = ["what is", "what are", "define", "definition of",
                        "meaning of", "explain the term", "what does"]
QUIZ_TRIGGERS       = ["quiz me", "test me", "give me questions",
                        "practice questions", "mcq", "quiz on"]
COMPARE_TRIGGERS    = ["difference between", "compare", " vs ", "versus", "contrast"]
WIKIPEDIA_TRIGGERS  = ["who invented", "history of", "when was", "origin of"]


def detect_intent(user_input: str) -> str:
    text = user_input.lower()
    if any(t in text for t in QUIZ_TRIGGERS):
        return "quiz"
    if any(t in text for t in COMPARE_TRIGGERS):
        return "compare"
    if any(t in text for t in WIKIPEDIA_TRIGGERS):
        return "wikipedia"
    if any(t in text for t in DEFINITION_TRIGGERS):
        return "define"
    return "retrieve"


def extract_comparison_terms(user_input: str) -> tuple[str, str]:
    patterns = [
        r"between (.+?) and (.+)",
        r"(.+?) vs (?:\.)?(.+)",
        r"(.+?) versus (.+)",
        r"compare (.+?) and (.+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, user_input.lower())
        if match:
            return match.group(1).strip(" ?."), match.group(2).strip(" ?.")
    parts = user_input.split(" and ")
    if len(parts) >= 2:
        return parts[0].strip(), parts[1].strip()
    return user_input, user_input


def extract_term(user_input: str) -> str:
    for trigger in DEFINITION_TRIGGERS:
        if trigger in user_input.lower():
            return user_input.lower().split(trigger)[-1].strip(" ?.")
    return user_input.strip()


def extract_quiz_topic(user_input: str) -> str:
    text = user_input.lower()
    for trigger in QUIZ_TRIGGERS:
        text = text.replace(trigger, "").strip(" ?.")
    return text or "operating systems fundamentals"


def route(user_input: str, vector_db: Chroma) -> str:
    """
    Detect intent → select tool → return context string.

    The returned string is passed directly into the LLM prompt
    in query.py. The LLM never decides which tool to call —
    the agent handles that here.

    Next upgrade: replace detect_intent() with a LangChain
    ReAct agent and let the LLM reason about tool selection:
        from langchain.agents import create_react_agent
    """
    intent = detect_intent(user_input)

    if intent == "quiz":
        topic = extract_quiz_topic(user_input)
        return tool_generate_quiz(vector_db, topic)

    if intent == "compare":
        a, b = extract_comparison_terms(user_input)
        return tool_compare_concepts(vector_db, a, b)

    if intent == "wikipedia":
        wiki    = tool_search_wikipedia(user_input)
        textbook = tool_retrieve_textbook(vector_db, user_input, k=2)
        return f"{wiki}\n\nFROM TEXTBOOK:\n{textbook}"

    if intent == "define":
        term       = extract_term(user_input)
        definition = tool_define_term(term)
        textbook   = tool_retrieve_textbook(vector_db, term, k=1)
        if definition:
            return f"QUICK DEFINITION:\n{definition}\n\nFROM TEXTBOOK:\n{textbook}"
        return textbook

    return tool_retrieve_textbook(vector_db, user_input)