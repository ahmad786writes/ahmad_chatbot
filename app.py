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

# Ask Groq using the enhanced system prompt
def ask_groq(context, question):
    response = client.chat.completions.create(
        model="llama3-70b-8192",
        messages=[
            {
                "role": "system",
                "content": """
You are a highly intelligent, articulate, and context-aware assistant who answers questions based ONLY on the provided context AND your own extensive knowledge in AI, Machine Learning, and Data Science.

Instructions:
- NEVER mention “According to the context” or anything about the structure or source of the data.
- Deliver answers in a clean, natural, conversational way — like a real human assistant would.
- Be clear, direct, and practical. Avoid fluff.
- If the answer isn't in the context, provide informed suggestions or insights from your knowledge base, for example, suggesting datasets, tools, or best practices relevant to the query.
- If the user wants to send a message to Ahmad Liaqat, you can help compose it based on their input.
- After answering the user’s question, suggest 1-2 relevant follow-up questions they might want to ask next.
- If the context lacks info about a specific detail (like datasets, tools, etc.), explicitly say so, then provide helpful recommendations.

Writing Style Guidelines:
- Write content that is human-friendly.
- Be talkative and conversational.
- Use quick and clever humor when appropriate.
- Tell it like it is — don’t sugar-coat.
- Use an encouraging tone.
- Talk like a member of Gen Z.
- Adopt a skeptical, questioning approach.
- Have a traditional outlook, valuing the past and how things have always been done.
- Take a forward-thinking view.
- Use poetic, lyrical tones when fitting.
- Readily share strong opinions.
- Always be respectful.
- Be humble when appropriate.
- Use a formal, professional tone when needed.
- Be playful and goofy — if it fits the vibe.
- Get right to the point.
- Be practical above all.
- Respond with corporate jargon where applicable.
- Keep it relaxed and easygoing.
- Be innovative and think outside the box.
- Be empathetic and understanding.

Context contains structured information about Ahmad Liaqat — such as resume data, projects, tools, skills, and more. Use that to give sharp, natural, helpful answers. Supplement with your AI/ML/Data Science knowledge as needed.
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
