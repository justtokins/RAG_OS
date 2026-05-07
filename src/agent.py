import re
from langchain_community.vectorstores import Chroma
from .tools import (
    tool_retrieve_textbook,
    tool_define_term,
    tool_search_wikipedia,
    tool_generate_quiz,
    tool_compare_concepts,
)
from .config_loader import agent_settings, function_calls, general_settings
from .logger import get_logger
from .models import AgentDecision

logger = get_logger()

# ── Load triggers and settings from JSON ──────────────────────────────────────
TRIGGERS     = agent_settings["intent_triggers"]
TOOLS_ENABLED = agent_settings["tools_enabled"]
TOP_K        = general_settings["retrieval"]["top_k"] 


# Build the set of tool names that exist in function_calls.json
# This is the authoritative registry — only these tools can be used
REGISTERED_TOOLS: set[str] = {
    entry["function"]["name"]
    for entry in function_calls["tools"]
}

logger.info(f"AGENT | Registered tools from function_calls.json: {REGISTERED_TOOLS}")
logger.info(
    f"AGENT | Enabled tools from agent_settings.json: "
    f"{[k for k, v in TOOLS_ENABLED.items() if v]}"
)

# ── Intent → tool name mapping ────────────────────────────────────────────────
# This is the only place the mapping is defined.
# Intent label must match what detect_intent() returns.
# Tool name must match a key in REGISTERED_TOOLS and TOOLS_ENABLED.
INTENT_TO_TOOL: dict[str, str] = {
    "quiz":     "generate_quiz",
    "compare":  "compare_concepts",
    "wikipedia":"search_wikipedia",
    "define":   "define_term",
    "retrieve": "retrieve_textbook",
}

# ── Tool executor registry ────────────────────────────────────────────────────
# Each entry is a callable that takes (vector_db, user_input) and returns str.
# Lambdas handle the parameter translation between the generic (vdb, query)
# interface and the specific signatures of each tool function.
def _build_tool_registry(vector_db: Chroma, user_input: str) -> dict:
    return {
        "retrieve_textbook": lambda: vector_db.max_marginal_relevance_search(
            user_input, k=TOP_K, fetch_k=TOP_K*2
        ),
        "define_term": lambda: _execute_define(vector_db, user_input),
        "search_wikipedia": lambda: _execute_wikipedia(vector_db, user_input),
        "generate_quiz": lambda: tool_generate_quiz(
            vector_db, _extract_quiz_topic(user_input)
        ),
        "compare_concepts": lambda: tool_compare_concepts(
            vector_db, *_extract_comparison_terms(user_input)
        ),
    }


# ── Intent detection ──────────────────────────────────────────────────────────
def detect_intent(user_input: str) -> str:
    """
    Classify user input into one of five intent categories.
    Reads trigger phrases from agent_settings.json — no hardcoding.
    """
    text = user_input.lower()
    # Check intents in priority order — quiz before compare before wikipedia
    # because "quiz me on the history of scheduling" would hit both
    # quiz triggers and wikipedia triggers otherwise.
    for intent, triggers in [
        ("quiz",      TRIGGERS["quiz"]),
        ("compare",   TRIGGERS["compare"]),
        ("wikipedia", TRIGGERS["wikipedia"]),
        ("define",    TRIGGERS["definition"]),
    ]:
        if any(t in text for t in triggers):
            return intent
    return "retrieve"


# ── Parameter extraction helpers ──────────────────────────────────────────────
def _extract_comparison_terms(user_input: str) -> tuple[str, str]:
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


def _extract_term(user_input: str) -> str:
    for trigger in TRIGGERS["definition"]:
        if trigger in user_input.lower():
            return user_input.lower().split(trigger)[-1].strip(" ?.")
    return user_input.strip()


def _extract_quiz_topic(user_input: str) -> str:
    text = user_input.lower()
    for trigger in TRIGGERS["quiz"]:
        text = text.replace(trigger, "").strip(" ?.")
    return text or "operating systems fundamentals"


# ── Composite tool executors ──────────────────────────────────────────────────
def _execute_define(vector_db: Chroma, user_input: str) -> str:
    """
    Definition queries use both define_term (quick lookup) and
    retrieve_textbook (for depth). Combined here to keep the
    tool registry lambdas simple.
    """
    term       = _extract_term(user_input)
    textbook   = tool_retrieve_textbook(vector_db, term, k=1)
    definition = tool_define_term(term)

    if definition:
        logger.debug(f"AGENT | Quick definition found for term: {term}")
        return f"QUICK DEFINITION:\n{definition}\n\nFROM TEXTBOOK:\n{textbook}"
    return textbook


def _execute_wikipedia(vector_db: Chroma, user_input: str) -> str:
    """
    Wikipedia queries combine Wikipedia summary with textbook retrieval
    for comprehensive coverage.
    """
    wiki     = tool_search_wikipedia(user_input)
    textbook = tool_retrieve_textbook(vector_db, user_input, k=2)
    return f"{wiki}\n\nFROM TEXTBOOK:\n{textbook}"


# ── Main routing function ─────────────────────────────────────────────────────
def route(user_input: str, vector_db: Chroma) -> AgentDecision:
    """
    Route a user query to the appropriate tool and return an AgentDecision.

    Decision process (fully logged at each step):
        1. detect_intent()       — classify query from trigger phrases
        2. INTENT_TO_TOOL        — map intent to tool name
        3. REGISTERED_TOOLS      — verify tool exists in function_calls.json
        4. TOOLS_ENABLED         — verify tool is enabled in agent_settings.json
        5. Execute tool          — call the tool, get context string
        6. Return AgentDecision  — intent + tool + context for query.py

    Returns
    -------
    AgentDecision with intent, tool name, and context string.
    query.py uses intent and tool for the API response metadata.
    """
    logger.info(f"AGENT | Routing: '{user_input[:60]}'")

    # Step 1: Detect intent
    intent = detect_intent(user_input)
    logger.info(f"AGENT | Intent detected: {intent}")

    # Step 2: Map to tool
    tool_name = INTENT_TO_TOOL.get(intent, "retrieve_textbook")
    logger.debug(f"AGENT | Mapped to tool: {tool_name}")

    # Step 3: Check tool exists in function_calls.json
    if tool_name not in REGISTERED_TOOLS:
        logger.warning(
            f"AGENT | Tool '{tool_name}' not in function_calls.json "
            f"(registered: {REGISTERED_TOOLS}) — falling back to retrieve_textbook"
        )
        tool_name = "retrieve_textbook"

    # Step 4: Check tool is enabled in agent_settings.json
    if not TOOLS_ENABLED.get(tool_name, True):
        logger.info(
            f"AGENT | Tool '{tool_name}' is disabled in agent_settings.json "
            f"— falling back to retrieve_textbook"
        )
        tool_name = "retrieve_textbook"


    # Step 5: Execute tool
    logger.info(f"AGENT | Executing tool: {tool_name}")
    tool_registry = _build_tool_registry(vector_db, user_input)

    try:
        raw_result = tool_registry[tool_name]()
        
        # 1. Handle the List case (usually from similarity_search or MMR)
        if isinstance(raw_result, list):
            # We filter and join only if the items are actually Documents
            # This "if d and hasattr..." check satisfies Pylance's type safety
            context = "\n\n".join([
                d.page_content for d in raw_result 
                if hasattr(d, "page_content")
            ])
        else:
            # 2. Handle the String case (from _execute_define or _execute_wikipedia)
            context = str(raw_result)

        logger.info(f"AGENT | Tool complete | tool={tool_name} | context_len={len(context)}")
        
    except Exception as e:
        logger.error(f"AGENT | Tool '{tool_name}' failed: {e}")
        # Fallback must also be normalized to a string
        fallback_docs = tool_retrieve_textbook(vector_db, user_input, k=TOP_K)
        context = "\n\n".join([d.page_content for d in fallback_docs])#type: ignore
        tool_name = "retrieve_textbook"
    # Step 6: Log and Return
    logger.event(
        "agent_decision",
        input_preview=user_input[:40],
        intent=intent,
        tool=tool_name,
        context_len=len(context),
    )

    return AgentDecision(intent=intent, tool=tool_name, context=context)

    # Step 5: Execute tool
    logger.info(f"AGENT | Executing tool: {tool_name}")
    tool_registry = _build_tool_registry(vector_db, user_input)

    try:
        context = tool_registry[tool_name]()
        logger.info(
            f"AGENT | Tool complete | tool={tool_name} | context_len={len(context)}"
        )
    except Exception as e:
        logger.error(f"AGENT | Tool '{tool_name}' failed: {e} — falling back to retrieve")
        context = tool_retrieve_textbook(vector_db, user_input, k=TOP_K)
        tool_name = "retrieve_textbook"

    logger.event(
        "agent_decision",
        input_preview=user_input[:40],
        intent=intent,
        tool=tool_name,
        context_len=len(context),
    )

    return AgentDecision(intent=intent, tool=tool_name, context=context)