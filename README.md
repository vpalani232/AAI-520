# 🤖 Multi-Agent Financial Analysis System

## Overview

This project develops a **Multi-Agent Financial Analysis System** designed to provide comprehensive, actionable investment insights. It integrates complex financial data retrieval, quantitative modeling, and synthesized news analysis using advanced AI architectures to simulate a team of financial researchers.

The system explores two primary methodologies for creating a robust, end-to-end research assistant.

---

## Key Features & Capabilities

* **End-to-End Investment Research:** Automatically generates structured reports combining technical analysis, fundamental metrics, and market sentiment.
* **Real-Time Quantitative Analysis:** Dedicated agent for fetching live stock data and performing calculations for:
    * **Technical Indicators:** RSI, MACD, Bollinger Bands, etc.
    * **Portfolio Optimization:** Modern Portfolio Theory (MPT), Sharpe Ratio, Value at Risk (VaR).
* **Advanced News Analysis (RAG):** Utilizes a **Retrieval-Augmented Generation (RAG)** pipeline, featuring:
    * Pre-processed historical news indexed in a **FAISS** vector database.
    * **Confidence Routing Workflow** for dynamic validation.
    * A **Confidence Checker Agent** that automatically triggers a fresh search (e.g., via SerpAPI) if the internal context relevance is low.
* **Hierarchical Agent Collaboration:** Uses the **CrewAI** framework to coordinate specialized agents under a central **Supervisor Agent**.

---

## Architectural Approaches

The system was developed using two distinct yet complementary agentic approaches:

### 1. Specialized Agentic Workflows
This approach focuses on building independent, highly-optimized workflows for specific tasks:
* **Financial News Agent:** Handles all aspects of news retrieval, RAG, confidence checks, and summarization using **NVIDIA NIM LLMs (Meta Llama-3.1)** for high-quality output.
* **Quantitative Financial Analyst Agent:** A tool-based agent capable of performing complex calculations, data visualization, and risk assessments.

### 2. Unified Multi-Agent Investment Research Crew (CrewAI)
A culmination of the first approach, this uses a **hierarchical process** where a `Supervisor Agent` orchestrates the entire research process:
* **Supervisor Agent:** Defines the overall goal and delegates sub-tasks.
* **Quantitative Analysis Agent:** Provides data-driven metrics.
* **News Agent:** Provides market sentiment and context.
* **Smart Summarizer Agent:** Synthesizes the output from the two specialized agents into a final, coherent investment report.

---

## Technologies Used

* **Frameworks:** Python, CrewAI
* **LLMs:** NVIDIA NIM (Meta Llama-3.1), potentially others (e.g., OpenAI models) configured via the environment.
* **Data/RAG:** FAISS (Vector Store), Custom tools for data fetching (e.g., financial APIs).
* **External Tools:** SerpAPI (for real-time web searches).

---

## Getting Started

1.  **Clone the repository:**
    ```bash
    git clone [Your Repository URL]
    cd Multi-Agent-Financial-Analysis
    ```
2.  **Set up the environment:**
    ```bash
    pip install -r requirements.txt
    ```
3.  **Configure API Keys in .env file:**
    ```
    NVIDIA_NIM_API_KEY=nvapi-XYZ
    SERPAPI_API_KEY=455cXXXXXXXXXXb
    ```
    * Set up necessary environment variables for your chosen LLM (e.g., `OPENAI_API_KEY`, `NIM_API_KEY`).
    * Configure API keys for data sources (e.g., financial data providers) and **SerpAPI**.
4.  **Run the analysis:**
    * Execute the `AAI_520_Team_10_Agentic_Financial_Research.ipynb` notebook cell by cell to explore both the independent and hierarchical agentic workflows.