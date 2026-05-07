"""
app.py — Streamlit frontend for SentiaVault AI.

Plug-and-play design:
    All backend URLs, feature toggles, labels, and colours live in
    frontend_config.json. Point this at any RAG backend by changing
    base_url in that file. No Python changes needed.

Background ingest flow:
    1. User uploads PDF → POST /upload returns job_id immediately
    2. st.rerun() loop polls GET /ingest/status/{job_id} every 2 seconds
    3. Spinner shows progress. On complete, success message appears.
    4. On failure, error is displayed with the reason from the backend.

Modular sections:
    Each sidebar section and main panel section is a function.
    To add a new feature, write a function and call it in the right place.
"""
import json
import time
from pathlib import Path

import requests
import streamlit as st

# ── Config loading ─────────────────────────────────────────────────────────────
CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "frontend_config.json"

@st.cache_data
def load_config() -> dict:
    with CONFIG_PATH.open(encoding="utf-8") as f:
        return json.load(f)
    
    

cfg = load_config()

APP      = cfg["app"]
BACKEND  = cfg["backend"]
CHAT     = cfg["chat"]
FEATURES = cfg["features"]
UPLOAD   = cfg["upload"]

BASE_URL = BACKEND["base_url"]
EP       = BACKEND["endpoints"]


# ── HTTP helpers ───────────────────────────────────────────────────────────────
def api(method: str, endpoint: str, **kwargs) -> dict | None:
    """Make a backend API call. Returns parsed JSON or None on failure."""
    url     = BASE_URL + endpoint
    timeout = BACKEND["request_timeout"]
    try:
        r = requests.request(method, url, timeout=timeout, **kwargs)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.ConnectionError:
        st.error("Cannot reach backend. Make sure `uvicorn main:app` is running.")
        return None
    except requests.exceptions.HTTPError as e:
        detail = ""
        try:
            detail = e.response.json().get("detail", "")
        except Exception:
            pass
        st.error(f"Backend error {e.response.status_code}: {detail or str(e)}")
        return None
    except Exception as e:
        st.error(f"Request failed: {e}")
        return None


# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title=APP["title"],
    page_icon=APP["logo_emoji"],
    layout="wide",
    initial_sidebar_state="expanded",
)


# ── Custom CSS ─────────────────────────────────────────────────────────────────
ACCENT = APP["accent_color"]

st.markdown(f"""
<style>
  /* ── Fonts ── */
  @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Syne:wght@400;600;800&display=swap');

  html, body, [class*="css"] {{
    font-family: 'Syne', sans-serif;
  }}
  code, pre, .stCode {{
    font-family: 'JetBrains Mono', monospace !important;
  }}

  /* ── Background ── */
  .stApp {{
    background: #0A0F1A;
    color: #E2E8F0;
  }}

  /* ── Sidebar ── */
  [data-testid="stSidebar"] {{
    background: #0D1424;
    border-right: 1px solid #1E2D45;
  }}
  [data-testid="stSidebar"] * {{
    color: #C5D0E6 !important;
  }}

  /* ── Header ── */
  .vault-header {{
    display: flex;
    align-items: center;
    gap: 14px;
    padding: 20px 0 8px;
    border-bottom: 1px solid #1E2D45;
    margin-bottom: 24px;
  }}
  .vault-logo {{
    font-size: 2.8rem;
    line-height: 1;
  }}
  .vault-title {{
    font-size: 2rem;
    font-weight: 800;
    color: #FFFFFF;
    letter-spacing: -0.03em;
    line-height: 1.1;
  }}
  .vault-subtitle {{
    font-size: 0.78rem;
    font-weight: 400;
    color: {ACCENT};
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-top: 2px;
  }}

  /* ── Chat messages ── */
  .stChatMessage {{
    background: #111827 !important;
    border: 1px solid #1E2D45;
    border-radius: 12px;
    padding: 4px 8px;
  }}

  /* ── Chat input ── */
  [data-testid="stChatInput"] {{
    background: #111827 !important;
    border: 1px solid #1E2D45 !important;
    border-radius: 12px !important;
    color: #E2E8F0 !important;
  }}
  [data-testid="stChatInput"]:focus {{
    border-color: {ACCENT} !important;
    box-shadow: 0 0 0 2px {ACCENT}33 !important;
  }}

  /* ── Buttons ── */
  .stButton > button {{
    background: {ACCENT} !important;
    color: #0A0F1A !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 600 !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 8px 18px !important;
    letter-spacing: 0.04em;
    transition: opacity 0.15s ease;
  }}
  .stButton > button:hover {{
    opacity: 0.85 !important;
  }}

  /* ── Expander ── */
  [data-testid="stExpander"] {{
    background: #0D1424 !important;
    border: 1px solid #1E2D45 !important;
    border-radius: 10px !important;
  }}

  /* ── Metric cards ── */
  [data-testid="stMetric"] {{
    background: #111827;
    border: 1px solid #1E2D45;
    border-radius: 10px;
    padding: 12px 16px;
  }}
  [data-testid="stMetricValue"] {{
    color: {ACCENT} !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 1rem !important;
  }}
  [data-testid="stMetricLabel"] {{
    color: #7A8FA6 !important;
    font-size: 0.72rem !important;
    text-transform: uppercase;
    letter-spacing: 0.08em;
  }}

  /* ── Status badges ── */
  .badge {{
    display: inline-block;
    padding: 2px 10px;
    border-radius: 20px;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
  }}
  .badge-complete  {{ background: #0D3D2A; color: #00D4AA; border: 1px solid #00D4AA44; }}
  .badge-processing{{ background: #2A1F00; color: #F59E0B; border: 1px solid #F59E0B44; }}
  .badge-failed    {{ background: #3D0D0D; color: #F87171; border: 1px solid #F8717144; }}
  .badge-queued    {{ background: #1A1F2E; color: #94A3B8; border: 1px solid #94A3B844; }}

  /* ── Intent / tool tags ── */
  .tag {{
    display: inline-block;
    padding: 2px 10px;
    border-radius: 6px;
    font-size: 0.72rem;
    font-family: 'JetBrains Mono', monospace;
    font-weight: 600;
    background: #1E2D45;
    color: {ACCENT};
    border: 1px solid {ACCENT}33;
    margin-right: 6px;
  }}

  /* ── Section labels ── */
  .section-label {{
    font-size: 0.65rem;
    font-weight: 600;
    color: #7A8FA6;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-bottom: 8px;
    border-bottom: 1px solid #1E2D45;
    padding-bottom: 4px;
  }}

  /* ── File uploader ── */
  [data-testid="stFileUploader"] {{
    background: #111827 !important;
    border: 1px dashed #1E2D45 !important;
    border-radius: 10px !important;
  }}

  /* ── Scrollbar ── */
  ::-webkit-scrollbar       {{ width: 5px; }}
  ::-webkit-scrollbar-track {{ background: #0A0F1A; }}
  ::-webkit-scrollbar-thumb {{ background: #1E2D45; border-radius: 4px; }}
  ::-webkit-scrollbar-thumb:hover {{ background: {ACCENT}66; }}
</style>
""", unsafe_allow_html=True)


# ── Session state initialisation ───────────────────────────────────────────────
def init_state():
    defaults = {
        "messages":      [],
        "session_id":    CHAT["default_session_id"],
        "ingest_job_id": None,
        "ingest_status": None,
        "books":         [],
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()


# ── Sidebar ────────────────────────────────────────────────────────────────────
def render_sidebar():
    with st.sidebar:
        # Logo
        st.markdown(f"""
        <div style="padding: 16px 0 20px">
          <div style="font-size:1.5rem; font-weight:800; color:#fff; letter-spacing:-0.02em">
            {APP['logo_emoji']} {APP['title']}
          </div>
          <div style="font-size:0.65rem; color:{ACCENT}; letter-spacing:0.12em; text-transform:uppercase; margin-top:2px">
            {APP['subtitle']}
          </div>
        </div>
        """, unsafe_allow_html=True)

        # ── Session ────────────────────────────────────────────────
        if FEATURES["show_session_input"]:
            st.markdown('<div class="section-label">Session</div>', unsafe_allow_html=True)
            new_id = st.text_input(
                "Session ID",
                value=st.session_state.session_id,
                label_visibility="collapsed",
                placeholder="session ID",
            )
            if new_id != st.session_state.session_id:
                st.session_state.session_id = new_id
                st.session_state.messages = []
                st.rerun()

        st.markdown("<hr style='border-color:#1E2D45; margin:12px 0'>", unsafe_allow_html=True)

        # ── Upload ─────────────────────────────────────────────────
        if FEATURES["show_upload"]:
            render_upload_section()

        st.markdown("<hr style='border-color:#1E2D45; margin:12px 0'>", unsafe_allow_html=True)

        # ── Books ──────────────────────────────────────────────────
        if FEATURES["show_books_list"]:
            render_books_section()

        st.markdown("<hr style='border-color:#1E2D45; margin:12px 0'>", unsafe_allow_html=True)

        # ── Health ─────────────────────────────────────────────────
        if FEATURES["show_health_check"]:
            render_health_section()


def render_upload_section():
    """Upload a PDF and start background ingestion with live status polling."""
    st.markdown('<div class="section-label">Knowledge Base</div>', unsafe_allow_html=True)

    uploaded = st.file_uploader(
        UPLOAD["label"],
        type=UPLOAD["accepted_types"],
        label_visibility="collapsed",
    )

    if st.button(UPLOAD["button_label"], use_container_width=True):
        if not uploaded:
            st.error("Select a PDF first.")
        else:
            with st.spinner(UPLOAD["encrypting_msg"]):
                resp = api(
                    "POST",
                    EP["upload"],
                    files={"file": (uploaded.name, uploaded.getvalue(), "application/pdf")},
                )
            if resp:
                st.session_state.ingest_job_id = resp["job_id"]
                st.session_state.ingest_status = "queued"
                st.rerun()

    # ── Ingest status polling ─────────────────────────────────
    if st.session_state.ingest_job_id:
        _poll_ingest_status()


def _poll_ingest_status():
    """
    Poll the backend for ingestion job status.
    Called on every Streamlit rerun while a job is active.
    When complete or failed, clears the job_id so polling stops.
    """
    job_id = st.session_state.ingest_job_id
    resp   = api("GET", f"{EP['ingest_status']}/{job_id}")

    if not resp:
        return

    status   = resp["status"]
    filename = resp.get("filename", "document")
    chunks   = resp.get("chunks", 0)
    error    = resp.get("error")

    badge_map = {
        "queued":     ("badge-queued",     "⏳ Queued"),
        "processing": ("badge-processing", "⚙ Processing"),
        "complete":   ("badge-complete",   "✓ Complete"),
        "failed":     ("badge-failed",     "✗ Failed"),
    }
    badge_cls, badge_label = badge_map.get(status, ("badge-queued", status))

    st.markdown(
        f'<span class="badge {badge_cls}">{badge_label}</span>'
        f' <span style="font-size:0.75rem; color:#7A8FA6">{filename}</span>',
        unsafe_allow_html=True,
    )

    if status == "complete":
        st.success(f"Indexed {chunks:,} chunks ✓")
        st.session_state.ingest_job_id = None
        st.session_state.ingest_status = "complete"
        # Refresh books list
        books_resp = api("GET", EP["books"])
        if books_resp:
            st.session_state.books = books_resp.get("books", [])

    elif status == "failed":
        st.error(f"Ingestion failed: {error}")
        st.session_state.ingest_job_id = None

    else:
        # Still running — wait then rerun to poll again
        time.sleep(BACKEND["poll_interval_ms"] / 1000)
        st.rerun()


def render_books_section():
    """Show all ingested documents."""
    st.markdown('<div class="section-label">Indexed Documents</div>', unsafe_allow_html=True)

    # Refresh on first load
    if not st.session_state.books:
        resp = api("GET", EP["books"])
        if resp:
            st.session_state.books = resp.get("books", [])

    books = st.session_state.books
    if books:
        for book in books:
            name    = book.get("filename", "Unknown")
            chunks  = book.get("chunks", 0)
            date    = (book.get("ingested_at") or "")[:10]
            st.markdown(
                f'<div style="padding:6px 0; border-bottom:1px solid #1E2D45">'
                f'  <div style="font-size:0.78rem; color:#E2E8F0; font-weight:600">{name}</div>'
                f'  <div style="font-size:0.68rem; color:#7A8FA6">{chunks:,} chunks · {date}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
    else:
        st.caption("No documents indexed yet.")


def render_health_section():
    """Show backend health status."""
    st.markdown('<div class="section-label">System Status</div>', unsafe_allow_html=True)

    if st.button("Check Status", use_container_width=True):
        resp = api("GET", EP["health"])
        if resp:
            status_icons = {
                "connected": "🟢",
                "loaded":    "🟢",
                "active":    "🟢",
                "error":     "🔴",
                "not loaded":"🟡",
            }
            for key, val in resp.items():
                icon = status_icons.get(val, "⚪")
                st.markdown(
                    f'<div style="display:flex;justify-content:space-between;'
                    f'padding:3px 0;font-size:0.75rem">'
                    f'  <span style="color:#7A8FA6;text-transform:capitalize">{key.replace("_"," ")}</span>'
                    f'  <span style="color:#E2E8F0">{icon} {val}</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )


# ── Main chat panel ────────────────────────────────────────────────────────────
def render_header():
    st.markdown(f"""
    <div class="vault-header">
      <div class="vault-logo">{APP['logo_emoji']}</div>
      <div>
        <div class="vault-title">{APP['title']}</div>
        <div class="vault-subtitle">{APP['subtitle']}</div>
      </div>
    </div>
    """, unsafe_allow_html=True)


def render_chat_history():
    """Render all past messages in the session."""
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            # If assistant message has metadata, show the expander
            if msg["role"] == "assistant" and msg.get("meta") and FEATURES["show_agent_logic"]:
                _render_agent_expander(msg["meta"])


def _render_agent_expander(meta: dict):
    """Render the 'Agent Intelligence' expander below an assistant message."""
    with st.expander("🧠 Agent Intelligence", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(
                f'<span style="font-size:0.72rem;color:#7A8FA6">INTENT</span><br>'
                f'<span class="tag">{meta.get("intent","—")}</span>',
                unsafe_allow_html=True,
            )
        with col2:
            st.markdown(
                f'<span style="font-size:0.72rem;color:#7A8FA6">TOOL</span><br>'
                f'<span class="tag">{meta.get("tool_used","—")}</span>',
                unsafe_allow_html=True,
            )


def render_welcome():
    """Show welcome message when chat is empty."""
    if not st.session_state.messages:
        st.markdown(f"""
        <div style="text-align:center; padding:60px 20px; color:#4A5568">
          <div style="font-size:3rem; margin-bottom:16px">{APP['logo_emoji']}</div>
          <div style="font-size:1.1rem; color:#7A8FA6; font-weight:400; max-width:480px; margin:0 auto; line-height:1.7">
            {CHAT['welcome_message']}
          </div>
        </div>
        """, unsafe_allow_html=True)


def render_chat_input():
    """Handle user input and backend call."""
    prompt = st.chat_input(CHAT["placeholder"])
    if not prompt:
        return

    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt, "meta": None})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Call backend
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            resp = api(
                "POST",
                EP["ask"],
                json={
                    "question":   prompt,
                    "session_id": st.session_state.session_id,
                    "format":     CHAT["output_format"],
                },
            )

        if resp:
            answer   = resp.get("answer", "No answer returned.")
            meta     = {
                "intent":   resp.get("intent",   "—"),
                "tool_used": resp.get("tool_used", "—"),
            }
            st.markdown(answer)

            if FEATURES["show_agent_logic"]:
                _render_agent_expander(meta)

            st.session_state.messages.append({
                "role":    "assistant",
                "content": answer,
                "meta":    meta,
            })
        else:
            error_msg = "⚠️ Could not reach the backend. Please try again."
            st.error(error_msg)
            st.session_state.messages.append({
                "role":    "assistant",
                "content": error_msg,
                "meta":    None,
            })


def render_stats_bar():
    """Show quick metrics row above the chat."""
    if not st.session_state.messages:
        return

    turns      = sum(1 for m in st.session_state.messages if m["role"] == "user")
    intents    = [m["meta"]["intent"] for m in st.session_state.messages
                  if m.get("meta") and m["meta"]]
    last_tool  = next(
        (m["meta"]["tool_used"] for m in reversed(st.session_state.messages)
         if m.get("meta") and m["meta"]), "—"
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Session",       st.session_state.session_id)
    c2.metric("Questions",     turns)
    c3.metric("Last Tool",     last_tool)
    c4.metric("Documents",     len(st.session_state.books))


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    render_sidebar()
    render_header()
    render_stats_bar()
    st.markdown("<hr style='border-color:#1E2D45; margin:0 0 16px'>", unsafe_allow_html=True)
    render_welcome()
    render_chat_history()
    render_chat_input()


if __name__ == "__main__":
    main()
