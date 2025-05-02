# main.py for FastAPI version of TrustGuardian

print("🚀 Starting FastAPI TrustGuardian...")

# --- 🔥 Import Necessary Libraries ---
# Import FastAPI and related items
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
import uvicorn
from pydantic import BaseModel # For defining expected input data
from typing import Optional, Dict, List, Any

# Import MOST libraries from your original app.py
# We might need to adjust these later if something is unused
import os, io, re, sys, json, numpy as np, time, fitz, tiktoken, traceback
from datetime import datetime
from langchain_groq import ChatGroq
from langchain.memory import ConversationSummaryBufferMemory
from langchain.schema import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_core.messages import HumanMessage, AIMessage
from langchain.chains import ConversationalRetrievalChain

print("✅ Base Libraries imported.")

# --- ⚙️ System Configuration & Globals (Copied from app.py) ---
print("⚙️ Configuring system settings...")
MAX_RETRIES = 3
DEBUG_MODE = True # Set to False for production if needed
VERSION = "2.0-FastAPI"
MEMORY_TOKENS = 2000
MAX_HISTORY_TOKENS = 4000
MAX_DOC_TOKENS_DIRECT = 3000
MAX_RAG_TOKENS = 4000

# --- Logger (Copied from app.py) ---
def log_debug(message: str) -> None:
    if DEBUG_MODE:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[DEBUG {timestamp}] {message}")

log_debug("Debug logging enabled.")

# --- Tokenizer Setup (Copied from app.py) ---
print("⚙️ Setting up tokenizer functions...")
def count_tokens_fallback(text: str) -> int: log_debug("Using estimated token count"); return len(text) // 4
def truncate_to_limit_fallback(text: str, max_tokens: int) -> str: log_debug("Using estimated truncation"); return text[:max_tokens*4]
count_tokens = count_tokens_fallback
truncate_to_limit = truncate_to_limit_fallback
try:
    token_manager = tiktoken.get_encoding("cl100k_base")
    def count_tokens_real(text: str) -> int:
        try: return len(token_manager.encode(text))
        except Exception as e: log_debug(f"Tiktoken count error: {e}. Falling back."); return count_tokens_fallback(text)
    def truncate_to_limit_real(text: str, max_tokens: int) -> str:
        try: tokens=token_manager.encode(text); T=tokens[:max_tokens] if len(tokens)>max_tokens else tokens; log_debug(f"Truncated tokens: {len(T)}/{len(tokens)}"); return token_manager.decode(T) # Corrected 'tok' to 'tokens'
        except Exception as e: log_debug(f"Tiktoken truncate error: {e}. Falling back."); return truncate_to_limit_fallback(text, max_tokens)
    count_tokens = count_tokens_real
    truncate_to_limit = truncate_to_limit_real
    print("✅ Tiktoken tokenizer functions ready.")
except Exception as e:
    print(f"⚠️ Warning: Failed tiktoken init: {e}. Using estimated token functions.")

# --- 🔑 Load API Keys ---
# IMPORTANT: In Vercel, these will be set as Environment Variables in the project settings.
# For local testing (like in Codespaces), you might need to set them manually if not already set.
print("🔐 Loading API keys from environment variables...")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")

if not GROQ_API_KEY or not PINECONE_API_KEY:
    warning_message = "⚠️ WARNING: GROQ_API_KEY or PINECONE_API_KEY not found in environment. App will likely fail."
    print(warning_message)
    # For Codespaces testing you might want to uncomment the line below and add keys temporarily
    # raise ValueError(warning_message) # Or handle this more gracefully
else:
    os.environ['PINECONE_API_KEY'] = PINECONE_API_KEY # Ensure Pinecone client sees it
    log_debug("API Keys retrieved from environment variables.")
    print("✅ API keys ready (found in environment).")

# --- ✨ Global Initializations (Create ONE instance of everything needed) ---
print("✨ Initializing core components (once at startup)...")
embedding_model = None
vectorstore = None
llm = None
memory = None
qa_chain = None
retriever = None
initialization_error = None

try:
    if GROQ_API_KEY and PINECONE_API_KEY: # Only initialize if keys are present
        log_debug("Initializing embedding model...")
        embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        print("✅ Embedding model ready.")

        log_debug("Setting up Pinecone vector store...")
        PINECONE_INDEX_NAME = "trustguardian" # Make sure this is correct
        vectorstore = PineconeVectorStore.from_existing_index(index_name=PINECONE_INDEX_NAME, embedding=embedding_model)
        log_debug(f"Connected to Pinecone index '{PINECONE_INDEX_NAME}'.")
        print("✅ Pinecone vector store ready.")

        log_debug("Initializing LLM...")
        llm = ChatGroq(groq_api_key=GROQ_API_KEY, model_name="llama-3.1-8b-instant") # Make sure model name is valid
        print(f"✅ LLM ready ({llm.model_name}).")

        log_debug("Setting up conversation memory...")
        # IMPORTANT: Memory needs careful handling in stateless APIs like FastAPI.
        # For simplicity now, we create one memory instance, but it won't remember across different API calls!
        # Proper session management is more complex. Let's start with this.
        memory = ConversationSummaryBufferMemory(llm=llm, max_token_limit=MEMORY_TOKENS, return_messages=True, memory_key="chat_history", output_key='answer')
        print("✅ Memory system ready (limited scope in API).")

        log_debug("Initializing Retriever...")
        retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 4, "fetch_k": 8, "lambda_mult": 0.5})
        print("✅ Retriever ready.")

        log_debug("Initializing ConversationalRetrievalChain...")
        qa_chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=retriever, memory=memory, return_source_documents=True, verbose=DEBUG_MODE)
        print("✅ ConversationalRetrievalChain ready.")
    else:
        initialization_error = "Missing API Keys, core components not initialized."
        print(f"🛑 {initialization_error}")

except Exception as e:
    initialization_error = f"Component initialization failed: {e}"
    log_debug(f"Initialization error: {e}\n{traceback.format_exc()}")
    print(f"🛑 FATAL ERROR: {initialization_error}")
    # We store the error but don't exit, so the API might partially start

# --- 📄 Document Processing Functions (Copied from app.py) ---
print("📄 Setting up document processing functions...")
class DocumentProcessor:
    @staticmethod
    def clean_text(t): log_debug("Cleaning(simplified)...");t=re.sub(r'\b(obj|endobj|stream|endstream|xref|trailer|startxref)\b','',t,flags=re.IGNORECASE);t=re.sub(r'\s+',' ',t).strip();return t
    @staticmethod
    def test_text_quality(t):
        log_debug(f"Testing quality(len:{len(t)})...");
        if not t or not t.strip():log_debug("Fail:Empty");return False,"Empty text"
        w=t.split();wc=len(w);uc=len(set(w));log_debug(f"W:{wc},U:{uc}")
        if wc<10:log_debug("Fail:W<10");return False,f"Too few words:{wc}"
        if uc<5:log_debug("Fail:U<5");return False,f"Too little variety:{uc}"
        log_debug("Pass.");return True,f"Quality OK:{wc} words"
    @staticmethod
    def extract_text_from_pdf(d):
        log_debug("Extracting(PyMuPDF)...");tp=[];doc=None
        try:
            doc=fitz.open(stream=d,filetype="pdf");[tp.append(p.get_text("text",sort=True))for i in range(len(doc))if(p:=doc.load_page(i))and p.get_text("text")]
            if doc: doc.close()
            full_text="\\n".join(filter(None, tp));log_debug(f"Extracted len:{len(full_text)}")
            return full_text
        except Exception as e: log_debug(f"PyMuPDF error:{e}"); raise ValueError(f"PyMuPDF failed:{e}")
        finally:
             if doc: doc.close()

def extract_text_from_uploaded_file(b):
     log_debug("\\n🔍 Processing upload...");t="";ct="";
     try:
         if not isinstance(b,bytes): raise ValueError("Expected bytes.")
         t=DocumentProcessor.extract_text_from_pdf(b);ct=DocumentProcessor.clean_text(t);
         q,m=DocumentProcessor.test_text_quality(ct);log_debug(f"Quality check:{m}")
         if not q:raise ValueError(f"Poor quality:{m}")
         return ct
     except Exception as e:err=f"Doc processing fail:{e}";log_debug(err);raise ValueError(err)
print("✅ Document processing functions ready.")

# --- Helper for Conditional Logic (Copied from app.py) ---
def query_seems_doc_specific(query: str) -> bool:
     query_lower=query.lower();dk=["this document","this file","uploaded document","uploaded file","summarize","summarise","analyze this","analyse this","extract from"]; is_s=any(k in query_lower for k in dk);log_debug(f"Query doc-specific check: {is_s}");return is_s

# --- 🧠 Main Application Class (Copied & adapted from app.py) ---
print("🔄 Setting up main application logic class...")
class TrustGuardian:
     # No __init__ needed if we use global components initialized above
     def handle_user_input(self, upload_data: Optional[bytes], user_query: str) -> str:
         log_debug(f"\\n🔄 Processing Request: '{user_query[:100]}...'"); text_to_return=""

         # Check if core components initialized properly
         if initialization_error:
             return f"⚠️ System Error: Core components failed to initialize. Reason: {initialization_error}"
         if not qa_chain or not llm or not memory: # Check essential components again
              return f"⚠️ System Error: Essential components (chain, llm, memory) are not ready."

         try:
             norm_q=user_query.lower().strip()
             if not norm_q: return "Please provide a question."
             if norm_q in["hi","hello","hey","salaam","salam","hola"]: return "👋 Hello! How can I assist you with compliance today?" # Simple greeting

             doc_is_uploaded=upload_data is not None
             is_doc_query=doc_is_uploaded and query_seems_doc_specific(user_query)

             if is_doc_query: # Mode 1: Doc-specific Query
                 log_debug("Mode: Doc Query - Direct LLM Call")
                 try:
                     doc_text=extract_text_from_uploaded_file(upload_data)
                     truncated_doc=truncate_to_limit(doc_text, MAX_DOC_TOKENS_DIRECT)
                     prompt=f"User Query:{user_query}\n\nDocument Content (may be truncated):\n{truncated_doc}\n\nInstructions: Answer the user's query based *only* on the provided Document Content above. Be concise and directly address the query."
                     log_debug(f"Doc-only prompt (~{count_tokens(prompt)} tokens)")
                     response_message=llm.invoke(prompt) # Use global llm
                     text_to_return=response_message.content.strip(); log_debug("Generated doc-specific response.")
                 except Exception as e: log_debug(f"Doc Proc/Query Error:{e}"); text_to_return=f"⚠️ Document Error: Could not process the document or query it. Details: {e}"

             else: # Mode 2: KB/Chat Query
                 log_debug("Mode: KB/Chat Query - Using Chain")
                 # NOTE: Memory handling limitation - history is reset per request here.
                 chat_history_messages = memory.chat_memory.messages
                 chain_input={"question":user_query,"chat_history":chat_history_messages} # Pass current (likely empty) history
                 log_debug(f"Invoking qa_chain w/ {len(chat_history_messages)} history msgs.")

                 result=qa_chain.invoke(chain_input) # Use global qa_chain

                 text_to_return=result.get("answer", "Sorry, I couldn't find an answer in my knowledge base.")

                 # --- Add Source Documents (if any) ---
                 source_docs = result.get("source_documents")
                 if source_docs:
                     citations = []
                     seen_sources = set()
                     for i, doc in enumerate(source_docs):
                         source_name = doc.metadata.get('source', f'Source {i+1}')
                         # Optional: Clean up source name if it's a long path/URL
                         source_name = os.path.basename(source_name) if source_name else source_name
                         if source_name not in seen_sources:
                             citations.append(f"📚 {source_name}")
                             seen_sources.add(source_name)

                     if citations:
                         text_to_return += "\n\n---\n**Sources:**\n" + "\n".join(citations)
                 # --- End Add Source Documents ---

                 # IMPORTANT: Clear memory AFTER the request for this simple stateless approach
                 # Otherwise, memory grows indefinitely on the server.
                 memory.clear()
                 log_debug("Memory cleared for next request.")


         except Exception as e:
             error_msg=f"Request processing error: {e}"
             log_debug(f"Error: {error_msg}\n{traceback.format_exc()}")
             text_to_return=f"⚠️ Error: An unexpected issue occurred while handling your request."

         return text_to_return if text_to_return else "Sorry, an unexpected issue occurred and no response was generated."

# --- Create ONE instance of the Guardian ---
guardian = TrustGuardian()
print("✅ Main application logic instance ready.")

# --- FastAPI App Definition ---
app = FastAPI(
    title="TrustGuardian API",
    description="API for compliance assistance. Ask questions or upload PDFs.",
    version=VERSION
)

# --- API Endpoints ---

# 1. Simple endpoint to check if the server is running
@app.get("/", tags=["Status"])
async def read_root():
    """Check if the API is running"""
    log_debug("GET / endpoint called")
    return {"message": f"TrustGuardian API v{VERSION} is running."}

# 2. The main endpoint to process queries (with optional file upload)
@app.post("/process", tags=["Processing"])
async def process_query_and_document(
    user_query: str = Form(...), # Get query from form data
    upload_file: Optional[UploadFile] = File(None) # Get optional file from form data
):
    """
    Process a user query, optionally using an uploaded PDF document.

    - **user_query**: The question you want to ask.
    - **upload_file**: (Optional) A PDF file to analyze with the query.
    """
    log_debug(f"POST /process called. Query: '{user_query[:50]}...', File uploaded: {upload_file is not None}")
    upload_bytes: Optional[bytes] = None
    filename: Optional[str] = None

    if upload_file:
        # Ensure it's a PDF before reading
        if upload_file.content_type != 'application/pdf':
             log_debug(f"Invalid file type: {upload_file.content_type}")
             raise HTTPException(status_code=400, detail="Invalid file type. Please upload a PDF.") # Use FastAPI's error helper

        filename = upload_file.filename
        log_debug(f"Reading uploaded file: {filename}")
        try:
            # Read the file content as bytes
            upload_bytes = await upload_file.read()
            log_debug(f"Read {len(upload_bytes)} bytes from file.")
        except Exception as e:
            log_debug(f"Error reading uploaded file: {e}")
            raise HTTPException(status_code=500, detail=f"Could not read uploaded file: {e}")
        finally:
             # Ensure the file handler is closed
             await upload_file.close()


    # Call the main handler logic from the guardian instance
    try:
        response_markdown = guardian.handle_user_input(upload_bytes, user_query)
        log_debug("Request processed successfully.")
        # Return the response as JSON
        return JSONResponse(content={"response": response_markdown})
    except HTTPException as e:
         # Re-raise HTTPExceptions directly
         raise e
    except Exception as e:
        log_debug(f"Error during handle_user_input call: {e}\n{traceback.format_exc()}")
        # Return a generic server error for other exceptions
        raise HTTPException(status_code=500, detail=f"An internal error occurred: {e}")

print("✅ FastAPI app defined with endpoints.")

# --- Run the app with Uvicorn (optional, Vercel will handle this) ---
# This part is useful if you want to test it locally in Codespaces
# Vercel uses its own way to run the app defined in vercel.json
# if __name__ == "__main__":
#     print("🚀 Launching Uvicorn server locally on 0.0.0.0:8000...")
#     # Run on port 8000 for local testing, Vercel uses a different port
#     uvicorn.run(app, host="0.0.0.0", port=8000)
#     # NOTE: You'd run this from the terminal using: uvicorn main:app --reload --port 8000

print("🏁 FastAPI TrustGuardian setup complete. Ready for Vercel deployment.")