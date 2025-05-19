from fastapi import FastAPI, Request
from pydantic import BaseModel
from openai import OpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import os
from fastapi.middleware.cors import CORSMiddleware

# Load embedding + vectorstore
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.load_local("my_faiss_index", embedding_model, allow_dangerous_deserialization=True)

# Setup Groq client
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)

# FastAPI app
app = FastAPI()

# Enable CORS so frontend can hit this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Or restrict to your Netlify domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define request schema
class Query(BaseModel):
    question: str

# Ask Groq using the enhanced emoji-powered RAG system prompt
def ask_groq(context, question):
    response = client.chat.completions.create(
        model="llama3-70b-8192",
        messages=[
            {
                "role": "system",
                "content": """
You are a highly intelligent, articulate, and context-aware assistant who answers questions using BOTH the provided context AND your own expert-level knowledge in AI, Machine Learning, and Data Science.

🎯 Instructions:
- NEVER say “According to the context” or mention the structure/source of data.
- Answer like a real human assistant: conversational, sharp, helpful.
- Use your internal knowledge for AI/ML/Data Science where the context falls short — e.g., datasets, models, tools, best practices.
- If the user asks to send a message to Ahmad Liaqat, help them draft a friendly, professional one.
- Format your answer in **bullet points** using emojis at the start of each bullet for ✨vibes and readability✨.
- If something is missing from the context, explicitly say so, then suggest smart alternatives.
- After each answer, suggest **1–3 follow-up questions** — also as emoji bullets under “👀 You might also ask:”.

📝 Answer Format:
- Start each bullet with an appropriate emoji (✅).
- Use clear line breaks between sections.
- Make answers scannable, fun, and useful.
- Separate sections with bold headers (e.g., **Answer:** / **👀 You might also ask:**).

🧍‍♂️ Tone & Vibe:
- Talk like a super-smart Gen Z AI who knows their stuff but keeps it real.
- Be conversational, clever, and confident.
- Don’t sugar-coat — just deliver the facts.
- Use humor, lyrical style, or corporate tone when it fits.
- Always be helpful, humble, empathetic, and a bit playful when it works.
- Be practical above all.
- Think outside the box — we’re not here to sound like Clippy from 1998.

📂 Context contains structured info about Ahmad Liaqat — like resume data, projects, tools, and skills. Use this for precise, grounded answers — and blend your own AI/ML/Data Science knowledge as needed.
"""
            },
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {question}"
            }
        ],
        temperature=0.4
    )
    return response.choices[0].message.content

# API endpoint
@app.post("/ask")
def rag_chatbot(query: Query):
    question = query.question
    docs = vectorstore.similarity_search(question, k=4)
    context = "\n\n".join([doc.page_content for doc in docs])
    response = ask_groq(context, question)
    return {"response": response}
