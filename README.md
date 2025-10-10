## AI-Powered SOC Workbench
## -Threat Analytics, ML & RAG (Streamlit)

This project implements a Retrieval-Augmented Generation (RAG) workflow to turn past incident data into actionable guidance. We first retrieve similar historical incidents to the analyst’s query using a TF-IDF cosine retriever (default, zero external dependencies). Optionally, you can enable semantic retrieval with Sentence-Transformers embeddings + FAISS for more robust matching across paraphrases. The retrieved incidents are surfaced in the UI and fed to a lightweight rule-based playbook generator, and can be piped into an LLM to produce an auto-generated, context-aware runbook.

