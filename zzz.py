from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uuid
import time
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain_ollama import OllamaLLM
from langchain.memory import ConversationBufferMemory
import re

# -------------------------------
# CONFIG
# -------------------------------
DB_DIR = "new_db"  # Chroma DB
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "llama3.2"
TOP_K = 3  # Docs to retrieve
SESSION_TTL = 6 * 60 * 60  # 6 hours in seconds

# -------------------------------
# PER-USER MEMORY
# -------------------------------
# session_id -> {"memory": ConversationBufferMemory(), "last_active": timestamp}
user_sessions = {}

# -------------------------------
# HELPER FUNCTIONS
# -------------------------------
def get_filter_from_question(question: str):
    q = question.lower()
    if re.search(r"\bleave\b|\bouting\b|\bweekend\b", q):
        return {"category": "VTOP"}
    elif re.search(r"\bplacement\b|\bpat\b", q):
        return {"category": "Placements"}
    elif "hostel" in q:
        return {"category": "Hostels"}
    elif "startup" in q:
        return {"category": "Startups"}
    else:
        return None

def load_retriever(filter_metadata: dict = None):
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    db = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)
    retriever = db.as_retriever(
        search_type="mmr",
        search_kwargs={"k": TOP_K, "fetch_k": 10}
    )
    if filter_metadata:
        retriever.search_kwargs["filter"] = filter_metadata
    return retriever

def build_prompt(context, question, chat_history):
    template = """
SYSTEM:
You are "CampusGuide" — a warm, human-like assistant for VIT-AP.

Rules:
- Small-talk → respond casually.
- Factual/general knowledge → answer directly.
- College-related questions → use context:
{context}
  * If context has steps, show them in order.
- If context doesn't contain the answer → respond naturally, suggest safe next step.
- Keep a friendly, student-to-student tone.

CHAT HISTORY:
{chat_history}

QUESTION:
{question}

Answer:
"""
    prompt = PromptTemplate(
        input_variables=["context", "question", "chat_history"],
        template=template
    )
    return prompt.format(context=context, question=question, chat_history=chat_history)

def run_query(query: str, memory) -> str:
    filter_metadata = get_filter_from_question(query)
    retriever = load_retriever(filter_metadata)
    llm = OllamaLLM(model=LLM_MODEL)

    docs = retriever.get_relevant_documents(query)
    context = "\n\n".join([d.page_content for d in docs]) if docs else ""

    history = "\n".join([f"{m.type}: {m.content}" for m in memory.chat_memory.messages])
    prompt = build_prompt(context, query, history)

    if not context.strip():
        greetings = ["hi", "hello", "hey", "hola", "how are you", "what's up"]
        if query.lower() in greetings:
            response = llm.invoke(f"Respond casually to: {query}")
        else:
            response = "Hmm, I don't see that in my data — you might want to check the official VIT-AP sources!"
    else:
        response = llm.invoke(prompt)

    memory.chat_memory.add_user_message(query)
    memory.chat_memory.add_ai_message(response)

    return response

# -------------------------------
# CLEAN OLD SESSIONS
# -------------------------------
def cleanup_sessions():
    now = time.time()
    to_delete = [sid for sid, info in user_sessions.items() if now - info["last_active"] > SESSION_TTL]
    for sid in to_delete:
        del user_sessions[sid]

# -------------------------------
# FASTAPI SETUP
# -------------------------------
app = FastAPI(title="CampusGuide API")

# Allow CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace with your frontend URL in production
    allow_methods=["*"],
    allow_headers=["*"],
)

class Question(BaseModel):
    question: str

@app.post("/ask")
def ask_question(q: Question, request: Request):
    cleanup_sessions()  # Remove old sessions automatically

    session_id = request.headers.get("X-Session-ID")
    clear_chat = request.headers.get("X-Clear-Chat") == "true"

    # If clear requested, delete the session completely
    if session_id in user_sessions and clear_chat:
        del user_sessions[session_id]
        # Optionally, return immediately if you want to acknowledge clear
        # return {"answer": "Chat cleared.", "session_id": None}

    # If no session ID, generate a new one
    if not session_id or session_id not in user_sessions:
        session_id = str(uuid.uuid4())
        user_sessions[session_id] = {
            "memory": ConversationBufferMemory(return_messages=True),
            "last_active": time.time()
        }

    memory_info = user_sessions[session_id]
    memory = memory_info["memory"]
    memory_info["last_active"] = time.time()  # update activity time

    answer = run_query(q.question, memory)
    return {"answer": answer, "session_id": session_id}


@app.get("/")
def root():
    return {"message": "CampusGuide API running"}
