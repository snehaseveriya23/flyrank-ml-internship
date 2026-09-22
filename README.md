# Google Search Ranking & Discoverability Intelligence (ML Capstone)

**Track**: Machine Learning | Week 8 Capstone  
**Lane**: Refresh / Content Opportunity Scoring  
**Data Source**: Built on the [FlyRank ML Internship dataset](https://flyrank.ai)  
**Deployed Paper**: `docs/index.html` (Accessible via GitHub Pages)

---

## 📖 Executive Summary & Deliverables

This repository contains the complete Machine Learning Capstone project for Google Search Ranking & Discoverability Intelligence:
1. **The Deployed Research Paper (`docs/index.html`)**: A publication-quality, visual research paper presenting the problem statement, leakage-safe methodology, Out-of-Time benchmark results, feature importances, limitations, and ranked action playbooks.
2. **Assignment & Capstone Notebooks (`work/`)**:
   - `01_data_exploration_and_prep.ipynb`: Exploratory data analysis, distribution profiling, and public-safe sanitization.
   - `02_feature_engineering_and_eda.ipynb`: Sliding lookback window feature engineering and signal correlation discovery.
   - `03_model_training_and_validation.ipynb`: Out-of-Time (OOT) holdout split validation benchmarking 4 model architectures.
   - `04_ranking_and_recommendation_engine.ipynb`: Capstone synthesis notebook generating prioritized editorial action queues with reason codes.
3. **Mandatory Submission Pointer (`submission/paper_url.txt`)**: Contains the single-line deployed research paper URL.
4. **End-to-End Pipeline (`work/capstone_pipeline.py`)**: Reproduces all datasets, trained models, benchmark JSONs, and chart visualizations.

---

## 📊 Key Results

| Model Architecture | ROC-AUC | PR-AUC (Avg Prec) | Brier Score | Precision @ Top 10% | Precision @ Top 20% |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Heuristic Baseline | 0.8253 | 0.6311 | 0.1645 | 0.8417 | 0.7417 |
| Logistic Regression | 0.9004 | 0.7709 | 0.1077 | 0.8667 | 0.8125 |
| Random Forest | 0.9199 | 0.8065 | 0.1020 | 0.8750 | 0.8292 |
| **Gradient Boosted Trees (Opportunity Engine)** | **0.9254** | **0.8172** | **0.0959** | **0.9000** | **0.8417** |

---

## 🚀 How to Run & Reproduce

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Full ML Pipeline
```bash
python work/capstone_pipeline.py
```

### 3. Deploy to GitHub Pages (Free & Instant)
1. Push this repository to your GitHub account (e.g. `https://github.com/<your-username>/google-search-ranking-capstone`).
2. Go to **Settings > Pages**.
3. Under **Build and deployment > Branch**, select `main` (or `master`) and folder `/docs`.
4. Click **Save**. Your research paper will be live at:
   `https://<your-username>.github.io/google-search-ranking-capstone/`
5. Update `submission/paper_url.txt` with your exact live URL.

---

## 🛡️ Public Safety & Data Credit
In strict compliance with public-safe rules, no confidential client identifiers, real domain names, or private query strings are included. All evaluations are performed on anonymized structures.

*Built on the FlyRank ML Internship dataset ([https://flyrank.ai](https://flyrank.ai)).*
