# RAG-chatbot

What it does:
→ Upload any PDF, DOCX, or TXT file
→ Ask questions in natural language
→ AI answers using ONLY your document content

Tech stack:
• LangChain for RAG pipeline
• FAISS vector database for semantic search
• sentence-transformers for local embeddings
• Groq (Llama 3.1) as the LLM
• Streamlit for the UI

The most interesting part was solving hallucination — 
by retrieving only relevant chunks before generating answers, 
the AI stays grounded in your actual documents.

Try it live 👇
[https://rag-chatbot-fe5ybpmekmpgxxlkh9hf3g.streamlit.app/]
