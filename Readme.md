# AI-Powered SOC Workbench — Threat Analytics, ML & RAG (Streamlit)

A production-minded Streamlit app for **cybersecurity incident analytics** with:
- 📊 Dashboards: trends, distributions, day×hour heatmaps, Pareto views
- 🤖 ML: severity classification, high/critical flag (optional), MTTR regression
- 🧠 RAG: retrieve similar incidents (TF-IDF baseline; optional embeddings + FAISS)
- 🧰 Playbooks: auto-suggested actions from retrieved incidents
- ☁️ One-click deployment: local, Cloudflare Tunnel, Render/Cloud Run

<p align="center">
  <img src="assets/screenshot-ui.png" alt="App UI screenshot" width="80%">
</p>

> Screenshot above is from the working app (replace with your own path).  
> Sample CSV expected: `merged_cyber_incidents.csv`.

---

## ✨ Features

- **Exploratory analytics**
  - Daily volume, Threat Type/Severity/Status bars
  - Incidents heatmap (Day of Week × Hour)
  - Top departments; High/Critical% by department
  - Financial impact hist & breach size vs. cost scatter

- **ML models**
  - Multiclass **severity** classifier (Logistic Regression)
  - Binary **high/critical** classifier (skips gracefully if only one class present)
  - **MTTR regression** (Ridge), trained when enough valid rows exist

- **RAG**
  - Baseline TF-IDF retriever (no external deps)
  - Optional **Sentence-Transformers + FAISS** for semantic search
  - Lightweight, rules-based **playbook generator** from nearest incidents

- **Easy deployment**
  - Local: `streamlit run`
  - Public link (dev): Cloudflare Tunnel
  - Managed: Render (free tier) or Cloud Run (serverless)

---

## 🗂️ Repository structure

