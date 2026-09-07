# RAG-Project
# This project is for German citizens and residents who can explore German-only government policy PDFs and get fast answers translated into # multiple languages.
/Library/Frameworks/Python.framework/Versions/3.10/bin/python3 -m venv rag_env source rag_env/bin/activate

pip install --upgrade pip #ollama pull qwen3:8b pip install langchain "langchain-community" "langchain-core" "langchain-ollama" "langchain_chroma" "sentence-transformers" pypdf "python-dotenv" dotenv "unstructured[pdf]" tiktoken gradio
