import wikipedia
from langchain_community.vectorstores import Chroma


# ── precompute term definitions (avoid recreation on every call)
_TERM_DEFINITIONS = {
    "process":        "A program in execution. The OS unit of work. "
                      "Like a running app on your phone.",
    "thread":         "A lightweight unit inside a process that shares "
                      "the same memory. Like multiple workers in one office.",
    "deadlock":       "Two or more processes waiting for each other forever — "
                      "none can proceed. Like two cars blocking each other "
                      "on a one-lane bridge.",
    "semaphore":      "A counter that controls access to a shared resource. "
                      "Like a bouncer enforcing a club's maximum capacity.",
    "mutex":          "A lock allowing only one thread at a time. "
                      "Like a single bathroom key in an office.",
    "paging":         "Dividing memory into fixed-size pages to eliminate "
                      "fragmentation. Like dividing a notebook into equal sections.",
    "scheduling":     "The OS deciding which process runs next on the CPU. "
                      "Like a restaurant manager deciding which order to cook first.",
    "context switch": "Saving one process's state and loading another's. "
                      "Like a chef pausing one dish to tend another.",
    "virtual memory": "Giving each process the illusion of a large private "
                      "memory space, backed by disk when RAM is full.",
    "inode":          "A data structure storing file metadata — size, owner, "
                      "disk location. Like a library card for a file.",
    "syscall":        "A controlled way for user programs to request OS services. "
                      "Like pressing a help button to call a supermarket employee.",
    "interrupt":      "A signal that pauses the CPU to handle an urgent event. "
                      "Like a fire alarm pausing a meeting.",
    "cache":          "Fast temporary storage holding recently used data. "
                      "Like keeping frequently used tools on your desk "
                      "instead of walking to the storeroom each time.",
}


# ── Tool 1 Textbook Retrieval
def tool_retrieve_textbook(vector_db: Chroma, query: str, k: int = 3) -> str:
    """
    Semantic search over the vector store.

    Embeds the query into the same 384-dim space as stored chunks,
    computes cosine similarity, returns the top-k passages.
    This is nearest-neighbour search in vector space.
    """
    docs = vector_db.similarity_search_with_score(query, k=k)

    if not docs:
        return "No relevant content found in the knowledge base."

    passages = []
    for doc, score in docs:
        page    = doc.metadata.get("page", "?")
        content = doc.page_content.strip()
        passages.append(f"[Page {page} | Relevance {round(1 - score, 3)}]\n{content}")

    return "\n\n".join(passages)


# ── Tool 2  Quick Term Definitions
def tool_define_term(term: str) -> str | None:
    """
    Returns a plain-English definition for common OS terms.
    Returns None if not found — caller falls back to retrieval.
    """
    return _TERM_DEFINITIONS.get(term.lower().strip())



# ── Tool 3  Wikipedia Fallback
def tool_search_wikipedia(query: str) -> str:
    """
    Fetch a short summary from Wikipedia.
    Used when the question goes beyond the uploaded textbook content.
    """
    try:
        summary = wikipedia.summary(query, sentences=3, auto_suggest=True)
        return f"[Wikipedia]\n{summary}"
    except wikipedia.exceptions.DisambiguationError as e:
        try:
            summary = wikipedia.summary(e.options[0], sentences=3)
            return f"[Wikipedia — {e.options[0]}]\n{summary}"
        except Exception:
            return "Wikipedia search returned ambiguous results."
    except wikipedia.exceptions.PageError:
        return "No Wikipedia page found for this query."
    except Exception as e:
        return f"Wikipedia search failed: {e}"


# ── Tool 4 Quiz Generator
def tool_generate_quiz(vector_db: Chroma, topic: str) -> str:
    """
    Retrieve content on a topic and structure a prompt asking
    the LLM to generate quiz questions from it.

    The tool handles retrieval. The LLM in query.py handles generation.
    Keeping these separate means the tool is reusable and testable
    without needing an LLM call.
    """
    content = tool_retrieve_textbook(vector_db, topic, k=2)
    return (
        f"QUIZ GENERATION REQUEST\n"
        f"Topic: {topic}\n"
        f"Source material from textbook:\n{content}\n\n"
        f"Generate 3 multiple-choice questions based strictly on this material. "
        f"Each question must have 4 options (A–D) with exactly one correct answer. "
        f"After each question explain why the correct answer is right."
    )


# ── Tool 5 Comparison
def tool_compare_concepts(vector_db: Chroma,
                           concept_a: str,
                           concept_b: str) -> str:
    """
    Retrieve textbook content on two concepts separately, then combine.
    Used for 'difference between X and Y' questions.
    """
    content_a = tool_retrieve_textbook(vector_db, concept_a, k=2)
    content_b = tool_retrieve_textbook(vector_db, concept_b, k=2)

    return (
        f"COMPARISON REQUEST\n\n"
        f"CONCEPT A — {concept_a.upper()}\n{content_a}\n\n"
        f"CONCEPT B — {concept_b.upper()}\n{content_b}\n\n"
        f"Using the content above, produce a clear comparison. "
        f"Highlight the key differences and when you would use each."
    )