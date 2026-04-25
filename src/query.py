"""
query.py — Async RAG pipeline.

All DB operations are now non-blocking:
    get_history()    — awaited  (needed before building prompt)
    save_message()   — fire-and-forget via asyncio.create_task()
                       (user already has the answer, no need to wait)

The LLM call (Groq) is the only truly blocking operation.
It runs in asyncio.to_thread() so the event loop stays free
to handle other requests while waiting for Groq to respond.
"""
import asyncio

from langchain_core.vectorstores import VectorStore
from groq import Groq

from database_pool import get_history, async_save_message
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
- If the answer is not in the context say so, then answer from general knowledge.
"""


def _call_groq(client: Groq, prompt: str) -> str:
    """
    Synchronous Groq call — wrapped in asyncio.to_thread() by bot().
    Defined as a plain sync function so it can be passed to to_thread().
    """
    response = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model=GROQ_MODEL,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
    )
    return response.choices[0].message.content or ""


async def bot(
    question:   str,
    vector_db:  VectorStore,
    client:     Groq,
    session_id: str = "default",
    fmt:        str = "markdown",
) -> dict:
    """
    Fully async RAG pipeline.

    Timeline:
        t=0   agent.route() starts (sync, fast — regex + vector search)
        t=0   get_history() starts concurrently via asyncio.gather()
        t=?   Both complete — whichever finishes last unblocks the gather
        t=?   Groq LLM call starts in thread pool (non-blocking)
        t=?   LLM responds
        t=?   Format answer
        t=?   Fire-and-forget: create_task(save user message)
        t=?   Fire-and-forget: create_task(save assistant message)
        t=?   Return immediately — DB writes finish in background

    The agent routing and history fetch run concurrently via gather().
    Neither depends on the other, so there is no reason to run them
    sequentially. This shaves off the history fetch latency from the
    critical path entirely.
    """
    logger.event("bot_start", session=session_id, q=question[:60])

    # ── Step 1: Agent routing + history fetch — concurrently ──────
    # asyncio.to_thread for route() because it calls vector_db.similarity_search
    # which may do I/O (e.g. Pinecone HTTP call). Safe to parallelise with history.
    decision_task = asyncio.to_thread(route, question, vector_db) # AgentDecision = intent + tool + context #type: ignore
    history_task  = get_history(session_id, limit=HISTORY_LIMIT)

    decision, history = await asyncio.gather(decision_task, history_task)

    logger.info(
        f"BOT | intent={decision.intent} tool={decision.tool} "
        f"history_turns={len(history)} context_len={len(decision.context)}"
    )

    # ── Step 2: Build prompt ───────────────────────────────────────
    prompt = build_prompt(question, decision.context, history)

    # ── Step 3: LLM call in thread (non-blocking) ──────────────────
    logger.info(f"BOT | Calling Groq {GROQ_MODEL}")
    try:
        raw_answer = await asyncio.to_thread(_call_groq, client, prompt)
    except Exception as e:
        logger.error(f"BOT | Groq call failed: {e}")
        raise

    logger.info(f"BOT | LLM responded | {len(raw_answer.split())} words")

    # ── Step 4: Format ─────────────────────────────────────────────
    formatted = formatter.format_answer(raw_answer, style=fmt)

    # ── Step 5: Save — fire and forget ────────────────────────────
    # create_task() schedules the coroutine on the event loop and returns
    # immediately. The response goes back to the user NOW. The DB writes
    # finish in the background, typically within a few milliseconds.
    asyncio.create_task(async_save_message(session_id, "user",      question))
    asyncio.create_task(async_save_message(session_id, "assistant", raw_answer))

    logger.event(
        "bot_complete",
        session=session_id,
        intent=decision.intent,
        tool=decision.tool,
    )

    return {
        "answer":    formatted,
        "intent":    decision.intent,
        "tool_used": decision.tool,
    }