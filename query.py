"""
query.py — RAG pipeline orchestration.

Flow for every request:
    1. agent.route()    → AgentDecision(intent, tool, context)
    2. get_history()    → decrypted conversation turns from PostgreSQL
    3. build_prompt()   → full prompt string
    4. Groq LLM call    → raw answer string
    5. formatter        → human-readable formatted output
    6. save_message()   → encrypted persistence of both turns
    7. return           → formatted answer + metadata

Nothing is initialised here. vector_db and client are injected
by main.py via FastAPI dependency injection.
"""
from langchain_community.vectorstores import Chroma
from groq import Groq

from database_pool import save_message, get_history
from agent import route
from formatter import OutputFormatter
from config_loader import general_settings
from logger import get_logger
from models import AgentDecision

logger    = get_logger()
formatter = OutputFormatter()

GROQ_MODEL    = general_settings["llm"]["model"]
TEMPERATURE   = general_settings["llm"]["temperature"]
MAX_TOKENS    = general_settings["llm"]["max_tokens"]
HISTORY_LIMIT = general_settings["retrieval"]["history_limit"]
PERSONA       = general_settings["persona"]


def build_prompt(question: str, context: str, history: list) -> str:
    """
    Assemble the full prompt sent to the LLM.

    Structure:
        [System identity]
        [Conversation history — last N turns]
        [Retrieved context from agent tool]
        [User question]
        [Instructions]

    Why include history?
        Without history, every question is stateless. The user cannot
        say "explain that more" because the LLM has no idea what "that"
        refers to. History gives the LLM memory of the conversation.

    Why put context before the question?
        LLMs attend more strongly to content near the question. Placing
        context just before the question improves retrieval utilisation.
    """
    history_text = "\n".join(
        f"{'User' if m['role'] == 'user' else PERSONA['name']}: {m['content']}"
        for m in history
    )

    return f"""You are {PERSONA['name']} — {PERSONA['role']}.
Tone: {PERSONA['tone']}.
Target audience: {PERSONA['target_level']}.

CONVERSATION HISTORY:
{history_text or 'No previous conversation.'}

CONTEXT (retrieved by agent):
{context}

QUESTION: {question}

INSTRUCTIONS:
- Use the context as your primary source.
- Define every technical term in plain English on first use.
- Give a real-world analogy for every concept.
- If context contains QUIZ GENERATION REQUEST → produce the quiz.
- If context contains COMPARISON REQUEST → produce a comparison table.
- If the answer is not in the context, say so clearly, then answer
  from general knowledge.
- Keep explanations clear, structured, and professional.
"""


def bot(
    question:   str,
    vector_db:  Chroma,
    client:     Groq,
    session_id: str = "default",
    fmt:        str = "markdown",
) -> dict:
    """
    Full RAG pipeline. Returns a dict so the route handler can
    include intent and tool_used in the API response.

    Parameters
    ----------
    question   : user's question string
    vector_db  : ChromaDB instance from app.state
    client     : Groq client from app.state
    session_id : conversation identifier
    fmt        : output format — 'markdown' | 'plain' | 'html'

    Returns
    -------
    {
        "answer":    formatted string,
        "intent":    detected intent label,
        "tool_used": tool name,
    }
    """
    logger.event("bot_start", session=session_id, question=question[:60])

    # ── Step 1: Agent routing ─────────────────────────────────────
    decision: AgentDecision = route(question, vector_db)
    logger.info(
        f"QUERY | intent={decision.intent} | tool={decision.tool} "
        f"| context_len={len(decision.context)}"
    )

    # ── Step 2: History ───────────────────────────────────────────
    history = get_history(session_id, limit=HISTORY_LIMIT)
    logger.debug(f"QUERY | history_turns={len(history)}")

    # ── Step 3: Prompt ────────────────────────────────────────────
    prompt = build_prompt(question, decision.context, history)
    logger.debug(f"QUERY | prompt_len={len(prompt)}")

    # ── Step 4: LLM call ──────────────────────────────────────────
    logger.info(f"QUERY | Calling Groq model={GROQ_MODEL}")
    try:
        response = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=GROQ_MODEL,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )
        raw_answer = response.choices[0].message.content or ""
        logger.info(f"QUERY | LLM responded | tokens={len(raw_answer.split())}")
    except Exception as e:
        logger.error(f"QUERY | LLM call failed: {e}")
        raise

    # ── Step 5: Format output ─────────────────────────────────────
    formatted_answer = formatter.format_answer(raw_answer, style=fmt)
    logger.debug(f"QUERY | Formatted answer | style={fmt}")

    # ── Step 6: Persist (encrypted) ───────────────────────────────
    save_message(session_id, "user",      question)
    save_message(session_id, "assistant", raw_answer)   # store raw, serve formatted
    logger.event("bot_complete", session=session_id, intent=decision.intent)

    return {
        "answer":    formatted_answer,
        "intent":    decision.intent,
        "tool_used": decision.tool,
    }