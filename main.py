# main.py
import os
import io
import requests
import asyncio
import hashlib
import tempfile
from typing import List, Dict, Optional
from fastapi import FastAPI, Request, HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# LangChain imports
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage

# --- Load Environment Variables ---
load_dotenv()

# --- API Key Validation ---
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
HACKRX_AUTH_TOKEN = os.getenv("HACKRX_AUTH_TOKEN", "c96eb4b747bdde4202dfb5ba2f1c91c5ef72eb741343a82c2194c4422d26754b") # From HackRX Guidelines

if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY environment variable is not set.")
if not HACKRX_AUTH_TOKEN:
    raise ValueError("HACKRX_AUTH_TOKEN environment variable is not set.")

# --- FastAPI App Initialization ---
app = FastAPI(
    title="HackRX GEN RAG API",
    description="Intelligent Query-Retrieval System.",
    version="1.0.0",
    docs_url="/api/v1/docs",
    redoc_url="/api/v1/redoc"
)

# --- Authentication Dependency ---
security = HTTPBearer()

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if credentials.scheme != "Bearer" or credentials.credentials != HACKRX_AUTH_TOKEN:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing authentication token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return credentials.credentials

# --- Initialize LLM and Embeddings (Global for reuse) ---
# Fix for asyncio RuntimeError on Streamlit/Render
try:
    _loop = asyncio.get_running_loop()
except RuntimeError:
    _loop = None
if _loop and _loop.is_running():
    pass
else:
    asyncio.set_event_loop(asyncio.new_event_loop())

# Initialize Embedding and LLM models once
embeddings = GoogleGenerativeAIEmbeddings(model="embedding-001")
llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro", temperature=0) # Use Gemini 2.5 Pro, low temp for factual answers

# In-memory cache for FAISS vector stores per document URL
# This avoids re-processing the same document multiple times within the same API instance.
app.state.vector_stores_cache: Dict[str, FAISS] = {}

# --- Document Processing Utilities ---
async def download_and_process_document(doc_url: str) -> List[Dict]:
    """Downloads document from URL and returns LangChain Document objects."""
    print(f"Attempting to download and process: {doc_url}")
    try:
        response = requests.get(doc_url, stream=True, timeout=15)
        response.raise_for_status()

        content_type = response.headers.get('Content-Type', '').lower()
        file_extension = doc_url.split('.')[-1].lower()

        if 'pdf' in content_type or file_extension == 'pdf':
            # --- CORRECT AND SOLE PDF PROCESSING BLOCK ---
            # Create a temporary file to save the PDF content
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                temp_file.write(response.content)
                temp_file_path = temp_file.name

            try:
                loader = PyPDFLoader(temp_file_path) # Pass the file path
                docs = loader.load()
                print(f"Successfully loaded {len(docs)} pages from PDF.")
                return docs
            finally:
                # Ensure the temporary file is deleted
                if os.path.exists(temp_file_path):
                    os.remove(temp_file_path)
            # --- END CORRECT PDF PROCESSING BLOCK ---
        # Add other loaders here if needed for DOCX/Email, following similar tempfile pattern
        # elif 'word' in content_type or file_extension == 'docx':
        #     # Example placeholder for DOCX - needs python-docx and a loader like Docx2txtLoader
        #     raise ValueError("DOCX support not fully implemented in this example.")
        # elif 'message/rfc822' in content_type or file_extension == 'eml':
        #     # Example placeholder for Email - needs email parsing library
        #     raise ValueError("Email parsing (EML) not fully implemented in this example.")
        else:
            # If no specific loader matches, raise error for unsupported type
            raise ValueError(f"Unsupported document type: {content_type} / .{file_extension}. Only PDFs are directly supported in this implementation.")

    except requests.exceptions.RequestException as e:
        print(f"Network or download error for {doc_url}: {e}")
        raise HTTPException(status_code=422, detail=f"Failed to download document from URL: {e}")
    except Exception as e:
        print(f"Error processing document {doc_url}: {e}")
        # Re-raising as HTTPException to ensure it's caught by FastAPI error handling
        raise HTTPException(status_code=422, detail=f"Error processing document content: {e}")

# --- Core RAG Functionality ---
async def get_or_create_vector_store(doc_url: str) -> FAISS:
    """
    Gets a FAISS vector store for a document URL from cache, or creates it.
    """
    doc_hash = hashlib.sha256(doc_url.encode()).hexdigest()

    if doc_hash in app.state.vector_stores_cache:
        print(f"Loading vector store from in-memory cache for {doc_url}")
        return app.state.vector_stores_cache[doc_hash]

    # Check if a persistent local copy exists (e.g., if deployed to a stateful service)
    # This path is mainly for local testing where /tmp/faiss_cache might persist across restarts
    FAISS_TEMP_PATH = f"/tmp/faiss_cache/{doc_hash}"
    if os.path.exists(FAISS_TEMP_PATH):
        print(f"Loading vector store from persistent disk cache for {doc_url}")
        try:
            faiss_db = FAISS.load_local(FAISS_TEMP_PATH, embeddings, allow_dangerous_deserialization=True)
            app.state.vector_stores_cache[doc_hash] = faiss_db
            return faiss_db
        except Exception as e:
            print(f"Failed to load from disk cache: {e}. Rebuilding...")
            # Fall through to dynamic build if pre-built fails to load


    print(f"Building new vector store for {doc_url}...")
    docs = await download_and_process_document(doc_url)
    if not docs:
        raise HTTPException(status_code=500, detail="Document processing failed, cannot build vector store.")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    text_chunks = text_splitter.split_documents(docs)

    faiss_db = FAISS.from_documents(text_chunks, embeddings)
    app.state.vector_stores_cache[doc_hash] = faiss_db

    # Optionally save to temp disk for future runs if container is persistent (e.g. non-free tier)
    os.makedirs(f"/tmp/faiss_cache", exist_ok=True)
    faiss_db.save_local(FAISS_TEMP_PATH)
    print(f"Vector store built and cached for {doc_url}.")
    return faiss_db

async def retrieve_answer(doc_url: str, question: str) -> str:
    """Retrieves relevant context and generates an answer.""" # Corrected "answ" to "answer"
    vector_store = await get_or_create_vector_store(doc_url)
    retriever = vector_store.as_retriever()

    # Prompt for RAG chain
    custom_prompt_template = """
    Use the pieces of information provided in the context to answer the user's question.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    Don't provide anything out of the given context.

    Question: {question}
    Context: {context}
    Answer:
    """
    rag_prompt = ChatPromptTemplate.from_template(custom_prompt_template)

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": rag_prompt}
    )

    response = qa_chain.invoke({"query": question})
    return response.get("result", "No relevant information found or unable to answer.")

# --- API Models ---
class RunRequest(BaseModel):
    documents: str = Field(..., description="URL to the document (PDF, DOCX, EML).")
    questions: List[str] = Field(..., description="List of natural language questions.")

class RunResponse(BaseModel): # Corrected 'Baseodel' to 'BaseModel' if it was still there
    answers: List[str] = Field(..., description="List of answers corresponding to the questions.")

# --- API Endpoint ---
@app.post("/api/v1/hackrx/run", response_model=RunResponse, status_code=status.HTTP_200_OK)
async def run_submission(
    request_body: RunRequest,
    auth_token: str = Depends(verify_token)
):
    print(f"Received request for document: {request_body.documents} with {len(request_body.questions)} questions.")
    answers = []
    for question in request_body.questions:
        try:
            answer = await retrieve_answer(request_body.documents, question)
            answers.append(answer)
        except HTTPException as e:
            answers.append(f"Error for question '{question}': {e.detail}")
        except Exception as e:
            answers.append(f"An unexpected error occurred for question '{question}': {e}")
    return RunResponse(answers=answers)

# --- Health Check Endpoint ---
@app.get("/health")
async def health_check():
    return {"status": "ok", "message": "API is healthy and ready to serve."}