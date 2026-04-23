import re
from langchain_community.vectorstores import Chroma
from tools import (
    tool_retrieve_textbook,
    tool_define_term,
    tool_search_wikipedia,
    tool_generate_quiz,
    tool_compare_concepts,
)
from config import agent_settings, general_settings

DEFINITION_TRIGGERS = agent_settings['intent_triggers']['definition']
QUIZ_TRIGGERS       = agent_settings['intent_triggers']['quiz']
COMPARE_TRIGGERS    = agent_settings['intent_triggers']['compare']
WIKIPEDIA_TRIGGERS  = agent_settings['intent_triggers']['wikipedia']
TOOLS_ENABLED       = agent_settings['tools_enabled']
TOP_K               = general_settings['retrieval']['top_k']


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
        if TOOLS_ENABLED.get("generate_quiz", True):
            return tool_generate_quiz(vector_db, topic)
        return tool_retrieve_textbook(vector_db, user_input, k=TOP_K)

    if intent == "compare":
        if TOOLS_ENABLED.get("compare_concepts", True):
            a, b = extract_comparison_terms(user_input)
            return tool_compare_concepts(vector_db, a, b)
        return tool_retrieve_textbook(vector_db, user_input, k=TOP_K)

    if intent == "wikipedia":
        if TOOLS_ENABLED.get("search_wikipedia", True):
            wiki    = tool_search_wikipedia(user_input)
            textbook = tool_retrieve_textbook(vector_db, user_input, k=TOP_K)
            return f"{wiki}\n\nFROM TEXTBOOK:\n{textbook}"
        return tool_retrieve_textbook(vector_db, user_input, k=TOP_K)

    if intent == "define":
        term = extract_term(user_input)
        textbook = tool_retrieve_textbook(vector_db, term, k=TOP_K)
        if TOOLS_ENABLED.get("define_term", True):
            definition = tool_define_term(term)
            if definition:
                return f"QUICK DEFINITION:\n{definition}\n\nFROM TEXTBOOK:\n{textbook}"
        return textbook

    return tool_retrieve_textbook(vector_db, user_input, k=TOP_K)