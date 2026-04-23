from langchain_community.vectorstores import Chroma
from groq import Groq
from database import save_message, get_history
from agent import route
from config_loader import general_settings
from logger import get_logger

GROQ_MODEL = general_settings['llm']['model']


def build_prompt(question: str, context: str, history: list) -> str:
    history_text = "\n".join(
        f"{'User' if m['role'] == 'user' else general_settings['persona']['name']}: {m['content']}"
        for m in history
    )
    return f"""You are {general_settings['persona']['name']} — {general_settings['persona']['role']}.
{general_settings['persona']['tone']}.
Target level: {general_settings['persona']['target_level']}.

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
    history  = get_history(session_id, limit=general_settings['retrieval']['history_limit'])
    prompt   = build_prompt(question, context, history)

    response = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model=GROQ_MODEL,
        temperature=general_settings['llm']['temperature'],
        max_tokens=general_settings['llm']['max_tokens'],
    )
    answer = response.choices[0].message.content
    if answer is None:
        answer = "I'm sorry, I couldn't generate an answer."

    save_message(session_id, "user",      question)
    save_message(session_id, "assistant", answer)

    return answer