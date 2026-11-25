# Website Q&A Bot (RAG)

Load 5-10 web pages with `WebBaseLoader`, split + embed them, and chat with a
RAG bot that shows the source URL for every answer.

## Run
```bash
python -m venv venv && source venv/bin/activate    # Windows: venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```
1. Pick a provider in the sidebar (Groq / Gemini need a free API key; Ollama needs `ollama pull llama3.2`).
2. Paste 5-10 URLs, click **Load & index pages**.
3. Ask questions. Sources appear under each answer.

## Files
- `app.py` - Streamlit UI
- `rag.py` - loading, chunking, embeddings (all-MiniLM-L6-v2), FAISS, prompt, LLM
- `.env.example` - API key template
