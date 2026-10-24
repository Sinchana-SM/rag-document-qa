# Intelligent Document Q&A (Local RAG)

Streamlit + LangChain + FAISS + Ollama (local chat model) + Sentence Transformers (local embeddings). No OpenAI API key or per-request API charge is needed. Models are downloaded once; the first download requires internet.

## Windows setup

1. Install Ollama for Windows from https://ollama.com/download/windows and open it.
2. In Command Prompt, download a small chat model:
   ```cmd
   ollama pull llama3.2:3b
   ```
   If that model is too slow or your computer has limited memory, try `ollama pull llama3.2:1b` and set `CHAT_MODEL=llama3.2:1b` in `.env`.
3. Open Command Prompt in this folder and activate your existing environment:
   ```cmd
   .venv\Scripts\activate.bat
   ```
4. Install the updated dependencies:
   ```cmd
   python -m pip install -r requirements.txt
   ```
5. Create `.env` from the example (optional; defaults are already set):
   ```cmd
   copy .env.example .env
   ```
6. Run the app:
   ```cmd
   streamlit run app.py
   ```
7. Upload PDF, DOCX, TXT or MD files, click **Build / update index**, then ask questions.

## Important
- First run downloads the embedding model from Hugging Face and may take a few minutes. It requires internet once; embeddings are then computed locally.
- Ollama must be running in the background. Test it with `ollama run llama3.2:3b`.
- If you previously created an index using OpenAI embeddings, click **Reset index** in the app before indexing again. Vector indexes made with different embedding models are not compatible.
- Models use disk space and local CPU/RAM. This is free of OpenAI API charges, not zero electricity or storage cost.

## Flow
upload -> load -> chunk + local embeddings -> FAISS -> retrieve relevant chunks -> local Ollama model answers from context -> UI shows sources.
