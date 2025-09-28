from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from rag_engine import CampusGuideRAG   # your existing RAG logic

# -------------------------------
# APP INIT
# -------------------------------
app = FastAPI(title="CampusGuide API")

# ✅ CORS (allow frontend to talk to backend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # or ["http://localhost:5173", "https://yourdomain.com"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------
# RAG ENGINE
# -------------------------------
rag = CampusGuideRAG()

# -------------------------------
# MODELS
# -------------------------------
class QueryRequest(BaseModel):
    question: str

# -------------------------------
# ROUTES
# -------------------------------
@app.get("/")
def root():
    return {"message": "CampusGuide is running 🚀"}

@app.post("/ask")
def ask_question(request: QueryRequest):
    answer = rag.run_query(request.question)
    return {"question": request.question, "answer": answer}
