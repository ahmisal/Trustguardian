# app.py for TrustGuardian Hugging Face Space

print("Starting TrustGuardian Application...")

# --- 🔥 Import Libraries ---
print("📚 Importing libraries...")
import os, io, re, sys, json, numpy as np, time, fitz, tiktoken, gradio as gr, traceback
from datetime import datetime
from typing import Optional, Dict, List, Any
from langchain_groq import ChatGroq
from langchain.memory import ConversationSummaryBufferMemory
from langchain.schema import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_core.messages import HumanMessage, AIMessage
from langchain.chains import ConversationalRetrievalChain
print("✅ Libraries imported.")

# --- ⚙️ System Configuration & Globals ---
print("\n⚙️ Configuring system settings...")
MAX_RETRIES = 3
DEBUG_MODE = True # Kept True as requested
VERSION = "2.0"
MEMORY_TOKENS = 2000
MAX_HISTORY_TOKENS = 4000
MAX_DOC_TOKENS_DIRECT = 3000 # Aggressive truncation for doc-only queries
MAX_RAG_TOKENS = 4000

# --- Logger ---
def log_debug(message: str) -> None:
    """Debug logger function"""
    if DEBUG_MODE:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[DEBUG {timestamp}] {message}")

log_debug("Debug logging enabled.")

# --- Tokenizer Setup (with robust fallback) ---
print("\n⚙️ Setting up tokenizer functions...")
# Define Fallback Functions FIRST
def count_tokens_fallback(text: str) -> int: log_debug("Using estimated token count"); return len(text) // 4
def truncate_to_limit_fallback(text: str, max_tokens: int) -> str: log_debug("Using estimated truncation"); return text[:max_tokens*4]
# Assign default functions
count_tokens = count_tokens_fallback
truncate_to_limit = truncate_to_limit_fallback
# Try to get real Tiktoken functions
try:
    token_manager = tiktoken.get_encoding("cl100k_base")
    def count_tokens_real(text: str) -> int:
        try: return len(token_manager.encode(text))
        except Exception as e: log_debug(f"Tiktoken count error: {e}. Falling back."); return count_tokens_fallback(text)
    def truncate_to_limit_real(text: str, max_tokens: int) -> str:
        try: tokens=token_manager.encode(text); T=tokens[:max_tokens] if len(tokens)>max_tokens else tokens; log_debug(f"Truncated tokens: {len(T)}/{len(tok)}"); return token_manager.decode(T)
        except Exception as e: log_debug(f"Tiktoken truncate error: {e}. Falling back."); return truncate_to_limit_fallback(text, max_tokens)
    # Overwrite the globals with the real functions
    count_tokens = count_tokens_real
    truncate_to_limit = truncate_to_limit_real
    print("✅ Tiktoken tokenizer functions ready.")
except Exception as e:
    print(f"⚠️ Warning: Failed tiktoken init: {e}. Using estimated token functions.")
# --- End Tokenizer Setup ---

# --- 🔑 Load API Keys from Environment Variables (Hugging Face Secrets) ---
print("\n🔐 Loading API keys from environment variables...")
GROQ_API_KEY = None
PINECONE_API_KEY = None
try:
    GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
    PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")

    if not GROQ_API_KEY: raise ValueError("Secret 'GROQ_API_KEY' not found.")
    if not PINECONE_API_KEY: raise ValueError("Secret 'PINECONE_API_KEY' not found.")

    # Set Pinecone key in environment for Langchain wrapper
    os.environ['PINECONE_API_KEY'] = PINECONE_API_KEY
    log_debug("API Keys retrieved from environment variables.")
    print("✅ API keys ready.")
except Exception as e:
    log_debug(f"Error loading API keys: {e}")
    print(f"FATAL ERROR: Could not load API keys from Secrets. Check Space settings. Error: {e}")
    sys.exit(1) # Exit if keys are missing

# --- Initializations (Embeddings, Pinecone, LLM, Memory, Chain) ---
print("\n✨ Initializing core components...")
embedding_model = None
vectorstore = None
llm = None
memory = None
qa_chain = None
retriever = None
try:
    log_debug("Initializing embedding model...")
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    print("✅ Embedding model ready.")

    log_debug("Setting up Pinecone vector store...")
    PINECONE_INDEX_NAME = "trustguardian"
    vectorstore = PineconeVectorStore.from_existing_index(index_name=PINECONE_INDEX_NAME, embedding=embedding_model)
    log_debug(f"Connected to Pinecone index '{PINECONE_INDEX_NAME}'.")
    print("✅ Pinecone vector store ready.")

    log_debug("Initializing LLM...")
    llm = ChatGroq(groq_api_key=GROQ_API_KEY, model_name="llama-3.1-8b-instant")
    print(f"✅ LLM ready ({llm.model_name}).")

    log_debug("Setting up conversation memory...")
    memory = ConversationSummaryBufferMemory(llm=llm, max_token_limit=MEMORY_TOKENS, return_messages=True, memory_key="chat_history", output_key='answer')
    print("✅ Memory systems ready.")

    log_debug("Initializing Retriever...")
    retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 4, "fetch_k": 8, "lambda_mult": 0.5})
    print("✅ Retriever ready.")

    log_debug("Initializing ConversationalRetrievalChain...")
    qa_chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=retriever, memory=memory, return_source_documents=True, verbose=DEBUG_MODE)
    print("✅ ConversationalRetrievalChain ready.")

except Exception as e:
    log_debug(f"Initialization error: {e}\n{traceback.format_exc()}")
    print(f"FATAL ERROR: Component initialization failed: {e}")
    sys.exit(1)
# --- End Initializations ---


# --- 📄 Document Processing Functions ---
print("\n📄 Setting up document processing functions...")
class DocumentProcessor: # ... (Class Definition as verified before) ...
    @staticmethod
    def clean_text(t): log_debug("Cleaning(simplified)...");t=re.sub(r'\b(obj|endobj|stream|endstream|xref|trailer|startxref)\b','',t,flags=re.IGNORECASE);t=re.sub(r'\s+',' ',t).strip();return t
    @staticmethod
    def test_text_quality(t): # Verified Multi-line formatting
        log_debug(f"Testing quality(len:{len(t)})...");
        if not t or not t.strip():log_debug("Fail:Empty");return False,"Empty text"
        w=t.split();wc=len(w);uc=len(set(w));log_debug(f"W:{wc},U:{uc}")
        if wc<10:log_debug("Fail:W<10");return False,f"Too few words:{wc}"
        if uc<5:log_debug("Fail:U<5");return False,f"Too little variety:{uc}"
        log_debug("Pass.");return True,f"Quality OK:{wc} words"
    @staticmethod
    def extract_text_from_pdf(d): # Using PyMuPDF
        log_debug("Extracting(PyMuPDF)...");tp=[];doc=None
        try:
            doc=fitz.open(stream=d,filetype="pdf");[tp.append(p.get_text("text",sort=True))for i in range(len(doc))if(p:=doc.load_page(i))and p.get_text("text")]
            if doc: doc.close() # Ensure close happens
            full_text="\\n".join(filter(None, tp));log_debug(f"Extracted len:{len(full_text)}")
            return full_text
        except Exception as e: log_debug(f"PyMuPDF error:{e}"); raise ValueError(f"PyMuPDF failed:{e}")
        finally:
             if doc: doc.close() # Extra safety close

def extract_text_from_uploaded_file(b): # ... (Definition as verified before) ...
     log_debug("\\n🔍 Processing upload...");t="";ct="";
     try:
         if not isinstance(b,bytes): raise ValueError("Expected bytes.")
         t=DocumentProcessor.extract_text_from_pdf(b);ct=DocumentProcessor.clean_text(t);
         q,m=DocumentProcessor.test_text_quality(ct);log_debug(f"Quality check:{m}")
         if not q:raise ValueError(f"Poor quality:{m}")
         return ct
     except Exception as e:err=f"Doc processing fail:{e}";log_debug(err);raise ValueError(err)
print("✅ Document processing functions ready.")

# --- Text Analysis Helpers (Optional) ---
# (Definitions as before - can be removed if not used by TrustGuardian logic)
def analyze_document_structure(t): log_debug("Analyzing doc structure (optional)..."); return {}
def extract_key_sections(t): log_debug("Extracting key sections (optional)..."); return []
print("✅ Text analysis helpers ready.")


# --- Helper for Conditional Logic ---
def query_seems_doc_specific(query: str) -> bool: # ... (Definition as verified before) ...
     query_lower=query.lower();dk=["this document","this file","uploaded document","uploaded file","summarize","summarise","analyze this","analyse this","extract from"]; is_s=any(k in query_lower for k in dk);log_debug(f"Query doc-specific check: {is_s}");return is_s

# --- 🧠 Main Application Class & Logic (Approach 1 - Conditional) ---
print("\n🔄 Setting up main application logic...")
class TrustGuardian: # ... (Definition using conditional logic and manual memory for chain as verified before) ...
     def __init__(self): log_debug("TrustGuardian initialized (conditional logic)")
     def handle_user_input(self, upload_data: Optional[bytes], user_query: str) -> str:
         log_debug(f"\\n🔄 Processing Request: '{user_query[:100]}...'"); text_to_return=""
         try:
             norm_q=user_query.lower().strip()
             if norm_q in["hi","hello","hey","salaam","salam","hola"]: return "👋 Hello! ..."
             doc_is_uploaded=upload_data is not None
             is_doc_query=doc_is_uploaded and query_seems_doc_specific(user_query)
             if is_doc_query: # Mode 1: Doc-specific Query
                 log_debug("Mode: Doc Query - Direct LLM Call")
                 try:
                     doc_text=extract_text_from_uploaded_file(upload_data)
                     truncated_doc=truncate_to_limit(doc_text, MAX_DOC_TOKENS_DIRECT)
                     prompt=f"User Query:{user_query}\n\nDocument Content(Truncated):\n{truncated_doc}\n\nInstructions:Answer based ONLY on doc."
                     log_debug(f"Doc-only prompt (~{count_tokens(prompt)} tokens)")
                     response_message=llm.invoke(prompt) # Use global llm
                     text_to_return=response_message.content.strip(); log_debug("Generated doc-specific response.")
                 except Exception as e: log_debug(f"Doc Proc/Query Error:{e}"); text_to_return=f"⚠️ Doc Error: {e}"
             else: # Mode 2: KB/Chat Query
                 log_debug("Mode: KB/Chat Query - Using Chain")
                 chat_history_messages = memory.chat_memory.messages # Use global memory
                 chain_input={"question":user_query,"chat_history":chat_history_messages}
                 log_debug(f"Invoking qa_chain w/ {len(chat_history_messages)} history msgs.")
                 result=qa_chain.invoke(chain_input) # Use global qa_chain
                 text_to_return=result.get("answer", "Sorry, no answer found.")
                 if result.get("source_documents"): # Append sources
                     citations=[f"📚 {doc.metadata.get('source',f'Src{i+1}')}" for i,doc in enumerate(result["source_documents"])]
                     if citations: text_to_return += "\n\n---\n📚 Sources:\n"+"\n".join(list(set(citations)))
         except Exception as e: error_msg=f"Request error:{e}";log_debug(f"Error:{error_msg}\n{traceback.format_exc()}");text_to_return=f"⚠️ Error:{error_msg}"
         return text_to_return if text_to_return else "Unexpected issue."
# --- End Class Definition ---

# --- Initialize Guardian Instance ---
guardian = TrustGuardian()
print("✅ Main application logic ready.")


# --- 🎨 Gradio Interface Definition ---
print("\n🎨 Setting up Gradio user interface...")
def ui_handler(upload_file_input, query): # ... (Definition as verified before) ...
    """Wrapper function for Gradio interface."""
    try:
        upload_bytes=None
        if upload_file_input is not None:
            if isinstance(upload_file_input,bytes):upload_bytes=upload_file_input;log_debug(f"Received {len(upload_bytes)} bytes.")
            else: log_debug(f"Warning: unexpected input type: {type(upload_file_input)}"); raise ValueError("Unexpected file type.")
        else: log_debug("No file uploaded.")
        if not isinstance(query,str): query=str(query) if query is not None else ""
        # Call main handler in the guardian instance
        response_markdown = guardian.handle_user_input(upload_bytes, query)
        return response_markdown
    except Exception as e: log_debug(f"Gradio Handler Error: {e}\n{traceback.format_exc()}"); return f"⚠️ System Error: {str(e)}"

# Define Gradio components
file_input=gr.File(label="📄 Upload Document (PDF)",type="binary",file_types=[".pdf"])
text_input=gr.Textbox(label="💭 Ask a Question",placeholder="E.g., 'Summarize doc' or 'HIPAA requirements?'",lines=3)
markdown_output=gr.Markdown(label="📝 Analysis & Response")

# Define the Interface
ui = gr.Interface(fn=ui_handler, inputs=[file_input, text_input], outputs=[markdown_output],
    title="🛡️ TrustGuardian – Compliance Assistant (v" + VERSION + ")",
    description="Upload PDF for analysis (uses first ~3000 tokens) or ask about GDPR, HIPAA, NIST, ISO 27001, SOC 2, PCI DSS.",
    allow_flagging="never")
print("✅ User interface defined.")


# --- Launch Gradio App ---
if __name__ == "__main__":
    print("\n🚀 Launching Gradio UI on 0.0.0.0:7860...")
    # Use server_name="0.0.0.0" for Docker/HF Spaces compatibility
    # Use server_port=7860 as default for HF Spaces
    # Keep debug=True as requested
    ui.launch(server_name="0.0.0.0", server_port=7860, debug=DEBUG_MODE)
    print(" Gradio launch sequence finished. Check logs for server status.")