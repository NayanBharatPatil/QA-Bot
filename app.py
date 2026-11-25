"""Streamlit UI for the Website Q&A Bot.  Run:  streamlit run app.py"""
import streamlit as st
from dotenv import load_dotenv

import rag

load_dotenv()

DEFAULT_URLS = """https://en.wikipedia.org/wiki/Python_(programming_language)
https://docs.python.org/3/tutorial/introduction.html
https://docs.python.org/3/tutorial/controlflow.html
https://docs.python.org/3/tutorial/datastructures.html
https://docs.python.org/3/tutorial/modules.html"""

st.set_page_config(page_title="Website Q&A Bot", page_icon="💬")
st.title("💬 Website Q&A Bot")
st.caption("Load 5-10 pages, then ask questions. Every answer shows its source URLs.")

with st.sidebar:
    st.header("1. LLM")
    provider = st.selectbox("Provider", ["Groq", "Gemini", "Ollama (local)"])
    api_key = ""
    if provider != "Ollama (local)":
        api_key = st.text_input(f"{provider} API key", type="password",
                                help="Or set it in a .env file")
    model = st.text_input("Model (optional)", placeholder="leave blank for default")

    st.header("2. Website pages")
    url_text = st.text_area("One URL per line (5-10)", DEFAULT_URLS, height=170)
    if st.button("Load & index pages", type="primary"):
        urls = list(dict.fromkeys(u.strip() for u in url_text.splitlines() if u.strip()))
        if not 5 <= len(urls) <= 10:
            st.error(f"Please enter 5-10 URLs (you entered {len(urls)}).")
        else:
            with st.spinner("Loading pages, splitting, embedding..."):
                try:
                    store, stats, failed = rag.build_index(urls)
                    st.session_state.update(store=store, messages=[], urls=urls)
                    st.success(f"Indexed {stats['pages']} pages "
                               f"({stats['chunks']} chunks).")
                    for u, err in failed:
                        st.warning(f"Skipped {u}: {err}")
                except Exception as exc:
                    st.error(str(exc))

if "store" not in st.session_state:
    st.info("👈 Add your LLM key and URLs in the sidebar, then click **Load & index pages**.")
    st.stop()

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m.get("sources"):
            with st.expander("Sources"):
                for s in m["sources"]:
                    st.markdown(f"**[{s['n']}]** {s['url']}  \n> {s['snippet']}")

if question := st.chat_input("Ask something about these pages..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        try:
            llm = rag.get_llm(provider, api_key, model)
            history = [(m["role"], m["content"]) for m in st.session_state.messages[:-1]]
            with st.spinner("Thinking..."):
                answer, sources = rag.ask(question, st.session_state.store, llm, history)
            st.markdown(answer)
            urls = list(dict.fromkeys(s["url"] for s in sources))
            st.markdown("**Source URLs:**\n" + "\n".join(f"- {u}" for u in urls))
            with st.expander("Retrieved passages"):
                for s in sources:
                    st.markdown(f"**[{s['n']}]** {s['url']}  \n> {s['snippet']}")
            st.session_state.messages.append(
                {"role": "assistant",
                 "content": answer + "\n\n**Source URLs:**\n" + "\n".join(f"- {u}" for u in urls),
                 "sources": sources})
        except Exception as exc:
            st.error(f"LLM error: {exc}")
