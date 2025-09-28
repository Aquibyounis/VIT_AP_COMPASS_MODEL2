from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain_ollama import OllamaLLM
from langchain.memory import ConversationBufferMemory


# -------------------------------
# CONFIG
# -------------------------------
DB_DIR = "chroma_db"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "llama3.2"    # Change if your Ollama model name differs
TOP_K = 3                 # Number of docs to retrieve


# -------------------------------
# MEMORY
# -------------------------------
memory = ConversationBufferMemory(return_messages=True)


class CampusGuideRAG:
    def __init__(self):
        self.retriever = self.load_retriever()
        self.llm = OllamaLLM(model=LLM_MODEL)

    def load_retriever(self):
        embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
        db = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)
        return db.as_retriever(
            search_type="mmr",
            search_kwargs={"k": TOP_K, "fetch_k": 10}
        )

    def build_prompt(self, context, question, chat_history):
        template = """
SYSTEM:
You are "CampusGuide" — a warm, natural, human-like assistant for VIT-AP.

Rules:
- If it's small-talk (hi, hello, how are you, what's up, hola, etc.) → respond casually, like a real person. Do NOT repeat "I'm CampusGuide" every time.
- If it's a factual or general knowledge question → answer directly, no need for context.
- If it's a college-related interrogative question → use this context:
{context}
  * If context has numbered steps, show them in order.
- If context doesn't contain the answer → say: "I don't see that in the info I have — here's a safe next step."
- Always keep a friendly, student-to-student tone. Short sentences, no stiff policy language.
- Use chat history to stay consistent and personal.

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

    def run_query(self, query: str) -> str:
        # Retrieve context
        docs = self.retriever.get_relevant_documents(query)
        context = "\n\n".join([d.page_content for d in docs]) if docs else ""

        # Get chat history
        history = "\n".join([f"{m.type}: {m.content}" for m in memory.chat_memory.messages])

        # Build prompt
        prompt = self.build_prompt(context, query, history)

        # Handle casual fallback
        if not context.strip() and query.lower() in ["hi", "hello", "hey", "hola", "how are you", "what's up"]:
            response = self.llm.invoke(f"Respond casually to: {query}")
        else:
            response = self.llm.invoke(prompt)

        # Save interaction
        memory.chat_memory.add_user_message(query)
        memory.chat_memory.add_ai_message(response)

        return response
