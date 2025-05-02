# TrustGuardian – GenAI Cyber Compliance Assistant

## Overview

TrustGuardian is a web application designed to help compliance officers, small business owners, and security auditors navigate the complex landscape of cybersecurity compliance. It leverages Generative AI (specifically Large Language Models via Groq) combined with Retrieval-Augmented Generation (RAG) using a Pinecone vector database to provide accurate answers to compliance questions and analyze uploaded compliance-related documents. This project aims to streamline compliance workflows and make expert guidance more accessible.

**Problem:** Staying compliant with multiple overlapping regulations (like GDPR, HIPAA, ISO 27001, NIST CSF, SOC 2, PCI DSS) is challenging, time-consuming, and critical for avoiding penalties and building trust.

**Solution:** This application provides an interactive interface where users can:
* Ask natural language questions about specific compliance standards.
* Upload their own documents (e.g., policies, reports) for analysis (like summarization or Q&A).
* Receive answers grounded in both a curated knowledge base and the content of their uploaded documents, with citations where applicable.

## Features

* **Conversational Q&A:** Ask questions about GDPR, HIPAA, ISO 27001, NIST CSF, and more, receiving contextual answers.
* **RAG Integration:** Answers to knowledge base questions are augmented with information retrieved from indexed compliance documents stored in Pinecone, improving accuracy and providing source references.
* **Document Analysis:** Upload PDF documents (policies, reports, etc.) and ask questions specifically about their content (e.g., "Summarize this document"). *Note: Analysis currently uses the first ~3000 tokens due to API limits.*
* **Compliance Frameworks Covered (in KB):** GDPR, HIPAA (including checklist), ISO 27001 (including control mapping), NIST CSF 2.0.
* **Interactive UI:** Simple web interface built with Gradio.

## Live Demo

You can access the live application deployed on Hugging Face Spaces here:

[**➡️ Access TrustGuardian Live Demo**](https://huggingface.co/spaces/ahmedsalman82/TrustGuardian-Cybersecurity.Compliance.Agent)

## Architecture

The application follows a RAG (Retrieval-Augmented Generation) pattern with conditional logic for handling different query types:

1.  **User Interface (Gradio):** Provides the web front-end for user interaction (text input, file upload). Hosted on Hugging Face Spaces.
2.  **Backend Logic (`app.py`):**
    * Receives input from Gradio.
    * **Conditional Routing:** Checks if a document is uploaded and if the query seems document-specific (using keywords).
    * **Path A (Doc Query):**
        * Uses `PyMuPDF` (`fitz`) to extract text from the uploaded PDF.
        * Performs basic text cleaning and quality checks.
        * Truncates extracted text (`~3000` tokens) to fit API limits.
        * Sends the query and truncated document text directly to the LLM via `langchain-groq`.
    * **Path B (KB/Chat Query):**
        * Uses `ConversationalRetrievalChain` from Langchain.
        * The chain uses `ConversationSummaryBufferMemory` to manage chat history.
        * It queries the `PineconeVectorStore` (via a Langchain Retriever) using embeddings (`sentence-transformers/all-MiniLM-L6-v2` via `langchain-huggingface`) to find relevant snippets from the indexed knowledge base (GDPR, HIPAA, etc.).
        * It sends the query, history, and retrieved KB snippets to the LLM (`llama-3.1-8b-instant` via `langchain-groq`).
        * The chain automatically updates the conversation memory.
3.  **Vector Database (Pinecone):** Stores embeddings of the curated compliance documents (knowledge base) for efficient retrieval. Index name: `trustguardian`.
4.  **LLM (Groq API):** Uses the `llama-3.1-8b-instant` model hosted on Groq for language understanding and generation.
5.  **Embeddings (Hugging Face):** Uses `sentence-transformers/all-MiniLM-L6-v2` via the `HuggingFaceEmbeddings` integration to create vector representations of text.

![Trustguardian drawio](https://github.com/user-attachments/assets/f2e955db-a258-4661-a6c4-b886c41333ac)


## Setup (Local Development/Testing)

Follow these steps to run the application locally:

1.  **Prerequisites:**
    * Python 3.9 or higher
    * Git

2.  **Clone Repository:**
    ```bash
    git clone [https://github.com/ahmisal/Trustguardian.git](https://github.com/ahmisal/Trustguardian.git)
    cd Trustguardian
    ```

3.  **Create Virtual Environment:**
    ```bash
    python -m venv venv
    # On Windows: venv\Scripts\activate
    # On macOS/Linux: source venv/bin/activate
    ```

4.  **Install Requirements:**
    ```bash
    pip install -r requirements.txt
    ```

5.  **Set Environment Variables:**
    * You need API keys for Groq and Pinecone.
    * Create a file named `.env` in the project root directory (this file is ignored by git via `.gitignore`).
    * Add your keys to the `.env` file like this:
        ```dotenv
        GROQ_API_KEY="gsk_YourActualGroqKey..."
        PINECONE_API_KEY="YourActualPineconeKey..."
        ```
    * The `app.py` script will load these using `os.environ.get()`. Ensure they are set correctly before running.

6.  **Run the Application:**
    ```bash
    python app.py
    ```
    The Gradio interface should launch on a local URL (e.g., `http://127.0.0.1:7860`). Note that the Pinecone index needs to be populated separately (see original Colab steps if needed for local testing).

## Hugging Face Deployment

This application is deployed on Hugging Face Spaces. The live deployment requires the following secrets to be set in the Space settings:

* `GROQ_API_KEY`
* `PINECONE_API_KEY`

The Space uses the `requirements.txt` file to build the environment and runs `app.py`.

## Knowledge Base Content

The Pinecone vector database (`trustguardian` index) has been populated with the text content derived from the following documents:

* GDPR (Full Text)
* HIPAA (Summary/Rule Text)
* HIPAA Compliance Checklist
* ISO 27001:2022 (Standard Text)
* ISO 27001 Control Mapping Example
* NIST CSF 2.0

## Known Limitations

* **Document Analysis Truncation:** Due to API rate limits on the free Groq tier, analysis/summarization of uploaded documents is based on the first ~3000 tokens only. Information beyond that point may be missed.
* **Conversation Summary:** The quality of LLM-generated summaries for the ongoing conversation can sometimes be inconsistent or misinterpret context.
* **Query Routing:** The logic deciding whether a query applies to an uploaded document or the knowledge base relies on simple keywords and may not handle ambiguous or complex comparative queries perfectly.
* **UI State:** The file upload input does not automatically clear after submission.

## License

This project is licensed under the **Apache 2.0 License**. See the `LICENSE` file for details.
