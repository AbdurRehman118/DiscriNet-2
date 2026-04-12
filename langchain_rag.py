"""
langchain_rag.py
────────────────
LangChain RAG Pipeline for Hate Speech Policy Explanation.

Architecture:
  1. Policies (JSONL)  →  GoogleGenerativeAIEmbeddings  →  FAISS VectorStore  (built once at startup)
  2. Input text        →  MMR Retriever (top-k=3)        →  retrieved policies
  3. Retrieved docs + input  →  Gemini LLM (fallback ladder)  →  natural-language explanation

Quota Optimisation:
  • LLM model fallback ladder:  gemini-2.0-flash → gemini-1.5-flash → gemini-1.5-flash-8b
  • @lru_cache(maxsize=128) on explanation function  →  zero API calls for repeated inputs
  • Skip LLM entirely if confidence < 0.15  →  static response for obvious non-hateful content
"""

import json
import os
from functools import lru_cache
from typing import List, Optional

# ── LangChain imports ──────────────────────────────────────────────────────────
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document

# ── Google API error types ─────────────────────────────────────────────────────
try:
    from google.api_core.exceptions import ResourceExhausted, ServiceUnavailable
except ImportError:
    # Fallback if google-api-core not directly accessible
    ResourceExhausted = Exception
    ServiceUnavailable = Exception

# ── Constants ──────────────────────────────────────────────────────────────────
# Model fallback ladder — tried in order when quota is exceeded
RAG_LLM_MODELS = [
    "gemini-flash-latest",        # Verified working on Key 3
    "gemini-2.0-flash",           # Fallback
    "gemini-1.5-flash-8b-latest", # Light fallback
]

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Skip LLM entirely if score is very low (clearly non-hateful)
SKIP_THRESHOLD = 0.15

# ── RAG Prompt Template ────────────────────────────────────────────────────────
RAG_PROMPT_TEMPLATE = """You are an expert moderator specialising in hate speech policy enforcement.
You have been given relevant policy documents and a piece of content to analyse.

=== RETRIEVED HATE SPEECH POLICIES ===
{context}

=== CONTENT TO ANALYSE ===
Text: "{question}"
Current Prediction: {prediction_label} (Confidence: {confidence:.1%})

=== YOUR TASK ===
Based ONLY on the retrieved policies above, provide a concise 2-3 sentence explanation.
You MUST strictly distinguish between the following categories in your response:
1. [HATEFUL]: Content that violates hate speech policies.
2. [UNWANTED / INAPPROPRIATE]: Content that is NSFW, Vulgar, or Gore (NOT Hate Speech).
3. [NON-HATEFUL / NORMAL]: Content that is Satire, Politics, Art, or everyday speech.

Cite specific policies by ID. If the content is "UNWANTED", explicitly state that it is safe from a hate-speech perspective but inappropriate for general audiences.

EXPLANATION:"""


# ── Module-level globals (built once at startup) ───────────────────────────────
_vectorstore: Optional[FAISS] = None
_retriever = None
_api_key: Optional[str] = None


def load_policies(policies_path: str) -> List[Document]:
    """Load JSONL policy file into LangChain Document objects with metadata."""
    docs: List[Document] = []
    with open(policies_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            # Store category in metadata for precise classification downstream
            docs.append(Document(
                page_content=obj.get("text", ""),
                metadata={
                    "id": obj.get("id", len(docs) + 1),
                    "category": obj.get("category", "hate")
                }
            ))
    if not docs:
        raise ValueError(f"No policies found in {policies_path}")
    print(f"[RAG] Loaded {len(docs)} policy documents.")
    return docs


def init_rag(api_key: str, policies_path: str, embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
    """
    Build the FAISS vector store from policy documents.
    """
    global _vectorstore, _retriever, _api_key

    _api_key = api_key

    print(f"[RAG] Initialising Local HuggingFace embeddings ({embedding_model})...")
    # This runs locally on CPU and bypasses all 403 API errors
    embeddings = HuggingFaceEmbeddings(
        model_name=embedding_model,
        model_kwargs={'device': 'cpu'}
    )

    docs = load_policies(policies_path)
    _vectorstore = FAISS.from_documents(docs, embeddings)

    # MMR retriever for diverse policy coverage
    _retriever = _vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 3, "fetch_k": 8, "lambda_mult": 0.7},
    )
    print("[RAG] FAISS vector store ready.")


def _retrieve_policies(text: str) -> List[Document]:
    """Retrieve top-3 most relevant policies for the given text."""
    if _retriever is None:
        raise RuntimeError("RAG not initialised. Call init_rag() first.")
    return _retriever.invoke(text)


def _build_prompt(text: str, prediction_label: str, confidence: float, context: str) -> str:
    """Format the RAG prompt with retrieved context."""
    return RAG_PROMPT_TEMPLATE.format(
        context=context,
        question=text,
        prediction_label=prediction_label,
        confidence=confidence,
    )


def _call_llm_with_fallback(prompt: str) -> str:
    """
    Call Gemini LLM with automatic fallback on quota errors.
    Tries models in order: gemini-2.0-flash → gemini-1.5-flash → gemini-1.5-flash-8b
    """
    last_error = ""
    for model_name in RAG_LLM_MODELS:
        try:
            print(f"[RAG] Trying LLM: {model_name}...")
            llm = ChatGoogleGenerativeAI(
                model=model_name,
                google_api_key=_api_key,
                temperature=0.2,
                max_output_tokens=1024, # Increased to prevent cutoffs
            )
            # Use StrOutputParser to safely handle all response types (lists, chunks, etc.)
            chain = llm | StrOutputParser()
            explanation = chain.invoke(prompt)
            
            print(f"[RAG] Success with {model_name}.")
            return explanation.strip()
        except (ResourceExhausted, ServiceUnavailable) as e:
            print(f"[RAG] {model_name} quota exceeded: {e}. Trying next model...")
            last_error = str(e)
            continue
        except Exception as e:
            # Non-quota error — propagate after logging
            err_str = str(e).lower()
            if "429" in err_str or "quota" in err_str or "resource_exhausted" in err_str:
                print(f"[RAG] {model_name} quota error (HTTP): {e}. Trying next model...")
                last_error = str(e)
                continue
            print(f"[RAG] Non-quota error with {model_name}: {e}")
            last_error = str(e)
            break

    return f"⚠️ RAG explanation unavailable (all models exhausted). Last error: {last_error[:80]}"


# ── Public API ─────────────────────────────────────────────────────────────────

@lru_cache(maxsize=128)
def get_rag_explanation(text: str, prediction_label: str, confidence: float) -> tuple:
    """
    Generate a policy-grounded LLM explanation for the given content.

    Returns:
        (explanation: str, retrieved_policies: list[str])

    Optimisations:
        - @lru_cache: identical (text, label, confidence) returns cached result instantly
        - SKIP_THRESHOLD: if confidence < 0.15, skip LLM entirely (clearly non-hateful)
    """
    # Early exit for obviously non-hateful content
    if confidence < SKIP_THRESHOLD:
        static_msg = (
            "Content is clearly non-hateful (confidence below 15%). "
            "No significant hate speech policy violations detected."
        )
        return static_msg, []

    # Retrieve relevant policies
    try:
        retrieved_docs = _retrieve_policies(text)
        # Format context with explicit labeling distinction for the LLM
        context_parts = []
        display_parts = []
        for doc in retrieved_docs:
            cat = doc.metadata.get("category", "unknown").upper()
            pid = doc.metadata.get("id", "?")
            txt = doc.page_content
            context_parts.append(f"Policy ID {pid} [CATEGORY: {cat}]: {txt}")
            display_parts.append(f"{pid}. [CAT: {cat}] {txt}")
            
        context = "\n\n".join(context_parts)
    except Exception as e:
        return f"⚠️ Policy retrieval failed: {e}", []

    # Build prompt and call LLM with fallback
    prompt = _build_prompt(text, prediction_label, confidence, context)
    explanation = _call_llm_with_fallback(prompt)

    return explanation, display_parts


def get_retrieved_policies_for_display(text: str) -> List[str]:
    """Lightweight retrieval-only call (no LLM) for displaying matched policies."""
    try:
        docs = _retrieve_policies(text)
        return [doc.page_content for doc in docs]
    except Exception:
        return []
