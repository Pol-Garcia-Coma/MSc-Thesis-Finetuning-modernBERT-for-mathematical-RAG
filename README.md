# Fine-Tuning ModernBERT for Mathematical Document Retrieval

This repository contains the code, data samples, and experiments developed as part of the Master’s thesis at the Barcelona School of Economics. The project explores how domain-specific fine-tuning of sentence embedding models can significantly enhance retrieval performance in Retrieval-Augmented Generation (RAG) systems, particularly for mathematical texts written in LaTeX.

We build a full pipeline that extracts structured math content (theorems, lemmas, definitions) from arXiv papers, generates synthetic queries using a large language model, and fine-tunes various ModernBERT variants using contrastive learning techniques. All experiments are designed to be reproducible in Google Colab, and models are loaded from and saved to private Hugging Face spaces.

---
## Authors
- Pol Garcia (pol.garcia@bse.eu)
- Natalia Lavrova (natalia.lavrova@bse.eu)
- Alex Malo (alex.malo@bse.eu)

##  Key Features

- **Contrastive Fine-Tuning**: Using Multiple Negatives Ranking Loss (MNRL), we train different versions of modernBERT to produce high-quality embeddings for retrieval.
- **Synthetic Query Generation**: We generate ~89,000 natural language questions using LLaMA 3.2 Instruct based on mathematical statements extracted from papers.
-  **Domain-Adaptive Pretraining (DAPT)**: We adapt the model to mathematical language using Masked Language Modeling on LaTeX-formatted arXiv papers.
- **Embedding Evaluation**: Performance is assessed with standard IR metrics (Accuracy@k, Recall@k, Precision@k, nDCG, MRR), showing significant improvements post fine-tuning.

---

##  Dataset Description

The dataset is derived from 12,000 mathematical research papers on arXiv (January–March 2025), scraped and parsed in their original LaTeX source format. From each paper, we extract formal mathematical statements (e.g., theorems, lemmas, definitions) using custom preprocessing techniques. Each extracted statement is paired with a synthetically generated question to form a (query, positive passage) training pair.

- Total pairs: **88,775**
- Total papers: **4,843**
- Median statements per paper: **13**
- Available in this repo: **a small sample of the dataset for inspection purposes**

---

##  How to Navigate the Repository

- `notebooks/` – Google Colab-compatible notebooks for preprocessing, fine-tuning, and evaluation.
- `scripts/` – Utility scripts for LaTeX parsing, segmentation, and data formatting.
- `config/` – Training and model configuration files.
- `data_sample/` – A sample of the (query, statement) dataset for testing and experimentation.
- `README.md` – Project documentation.
- `LICENSE` – Licensing information.

