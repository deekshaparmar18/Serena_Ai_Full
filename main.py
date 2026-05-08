from fastapi import FastAPI, UploadFile, File
from pypdf import PdfReader

# LangChain (updated imports)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import os

# Gemini (new SDK)
from google import genai

app = FastAPI()
#  API KEY 
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# Global vector DB
vector_store = None


#  1. Upload PDF API
@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    global vector_store

    try:
        # Read PDF
        content = await file.read()
        reader = PdfReader(file.file)

        text = ""
        for page in reader.pages:
            content = page.extract_text()
            if content:
                text += content

        if not text:
            return {"error": "No text found in PDF"}

        # Split text
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )

        chunks = splitter.split_text(text)

        # Embeddings
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        # Create vector DB
        vector_store = FAISS.from_texts(chunks, embeddings)

        return {"message": "PDF uploaded & processed successfully"}

    except Exception as e:
        print("UPLOAD ERROR:", e)
        return {"error": str(e)}


# 2. Ask Question API
@app.get("/ask")
def ask_question(q: str):
    global vector_store

    if vector_store is None:
        return {"error": "Upload PDF first"}

    try:
        # Search similar chunks
        docs = vector_store.similarity_search(q)

        if not docs:
            return {"error": "No relevant content found"}

        # Build context
        context = ""
        for d in docs:
            context += d.page_content + "\n"

        # Prompt
        prompt = f"""
        Answer the question based only on the context below.

        Context:
        {context}

        Question:
        {q}
        """

        # Gemini call
        response = client.models.generate_content(
            model="gemini-2.5-flash", 
            contents=prompt
        )

        return {
            "question": q,
            "answer": response.text
        }

    except Exception as e:
        print("ASK ERROR:", e)
        return {"error": str(e)}