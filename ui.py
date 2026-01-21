import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000/run"

st.set_page_config(
    page_title="Multi-Agent Research System",
    page_icon="🧠",
    layout="wide"
)

# ---------- GLOBAL CSS (Dark Gradient + Glass UI) ----------
st.markdown("""
<style>

html, body {
    background: radial-gradient(circle at top, #1f1f1f 0%, #0d0d0d 100%);
    color: white !important;
}

textarea {
    border-radius: 12px !important;
    background-color: #2a2a2a !important;
    color: white !important;
    font-size: 15px !important;
}

.card {
    background: rgba(255, 255, 255, 0.08);
    padding: 18px;
    border-radius: 12px;
    border: 1px solid rgba(255,255,255,0.12);
    backdrop-filter: blur(10px);
    -webkit-backdrop-filter: blur(10px);
    margin-bottom: 12px;
    font-size: 16px;
}

.title {
    font-size: 22px;
    font-weight: 600;
    margin-bottom: 6px;
    color: #f7d774;
}

.run-btn {
    width: 180px;
    background: linear-gradient(90deg, #4b7bec, #3867d6);
    border-radius: 10px;
    padding: 10px;
    border: none;
    font-weight: 600;
    font-size: 16px;
}

</style>
""", unsafe_allow_html=True)


# ---------- HEADER ----------
st.markdown(
    "<h2 style='text-align:center; font-family:Verdana;'>AI_MULTI_AGENT</h2>"
    "<p style='text-align:center; color:#ccc; font-size:14px;'>Gemini + Wikipedia + WebSearch + Arxiv</p>",
    unsafe_allow_html=True
)


# ---------- INPUT ----------
col1, col2, col3 = st.columns([1,2,1])
with col2:
    query = st.text_area(
        "Enter your query:",
        height=110,
        placeholder="Ex: Research blockchain, summarize quantum computing, email about AI in healthcare"
    )

with col2:
    run = st.button("Run Agents", use_container_width=True)


# ---------- QUIT IF EMPTY ----------
if run and not query.strip():
    st.warning("Please enter a valid query")
    st.stop()


# ---------- CALL BACKEND ----------
if run:
    with st.spinner("AI Agents Working..."):
        resp = requests.post(API_URL, json={"query": query})
        if resp.status_code != 200:
            st.error("Backend error occurred")
            st.write(resp.text)
            st.stop()

        data = resp.json()

    # ---------- TABS LAYOUT ----------
    tab1, tab2, tab3 = st.tabs(["📚 Research", "✍️ Summary", "✉️ Email"])

    # -------- TAB 1: RESEARCH --------
    with tab1:
        research = data.get("research")
        if research:
            wiki = research.get("wiki_web")
            arxiv = research.get("arxiv")

            if wiki:
                st.markdown("<div class='card'><div class='title'>Wikipedia + Web Results</div>" + wiki + "</div>", unsafe_allow_html=True)

            if arxiv:
                st.markdown("<div class='card'><div class='title'>Arxiv (Top-5 Papers)</div>" + arxiv + "</div>", unsafe_allow_html=True)

    # -------- TAB 2: SUMMARY --------
    with tab2:
        summary = data.get("summary")
        if summary:
            st.markdown("<div class='card'><div class='title'>Structured Summary</div>" + summary + "</div>", unsafe_allow_html=True)

    # -------- TAB 3: EMAIL --------
    with tab3:
        email = data.get("email")
        if email:
            email = email.replace("\n", "<br>")
            st.markdown("<div class='card'><div class='title'>Formal Email</div>" + email + "</div>", unsafe_allow_html=True)