from langchain_community.vectorstores import Chroma
from groq import Groq
from database import save_message, get_history
from agent import route

GROQ_MODEL = "llama-3.1-8b-instant"


def build_prompt(question: str, context: str, history: list) -> str:
    history_text = "\n".join(
        f"{'User' if m['role'] == 'user' else 'OS_Helper'}: {m['content']}"
        for m in history
    )
    return f"""You are OS_Helper — an expert Operating Systems assistant.
Explain every concept clearly with real-world analogies.
Target level: senior high school student.

CONVERSATION HISTORY:
{history_text or 'No previous conversation.'}

CONTEXT (retrieved by the agent):
{context}

QUESTION: {question}

INSTRUCTIONS:
- Use the context as your primary source.
- Define every technical term in plain English.
- Give a real-world analogy for every concept.
- If context contains a QUIZ GENERATION REQUEST, produce the quiz.
- If context contains a COMPARISON REQUEST, produce a clear comparison table.
- If the answer is not in the context, say so then answer from general knowledge.
- Keep explanations clear, structured, and professional.
"""


def bot(question: str,
        vector_db: Chroma,
        client: Groq,
        session_id: str = "default") -> str:
    """
    Full RAG pipeline:
        1. Agent routes question to the right tool
        2. Tool retrieves relevant context
        3. Prompt is built with context + history
        4. Groq LLM generates the answer
        5. Exchange is saved to PostgreSQL

    Parameters are passed in — nothing is initialised here.
    All setup happens in main.py lifespan and is injected via app.state.
    """
    context  = route(question, vector_db)
    history  = get_history(session_id, limit=6)
    prompt   = build_prompt(question, context, history)

    response = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model=GROQ_MODEL,
        temperature=0.3,
        max_tokens=1024,
    )
    answer = response.choices[0].message.content

    save_message(session_id, "user",      question)
    save_message(session_id, "assistant", answer)

    return answer