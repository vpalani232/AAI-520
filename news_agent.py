# ============================================================
# NEWS ANALYSIS AGENT & TASK
# ============================================================

from crewai import Agent, Task, Crew, LLM
from crewai.tools import tool
from datetime import datetime
import json
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import pandas as pd
import numpy as np
import requests
import os
from datetime import datetime
from serpapi import GoogleSearch

from typing import Tuple, List, Dict, Any

import os
from dotenv import load_dotenv

try:
    from serpapi import GoogleSearch
except ImportError:
    # Define a mock class for demonstration if serpapi is not installed
    class GoogleSearch:
        def __init__(self, params):
            pass

        def get_dict(self):
            return {"news_results": []}

    print("Warning: 'serpapi' library not found. Using a mock class.")

# Load variables from .env file
load_dotenv()

NIM_ENDPOINT = "https://integrate.api.nvidia.com/v1/chat/completions"
NIM_MODEL = "meta/llama-3.1-405b-instruct"

llm = LLM(model="nvidia_nim/meta/llama-3.1-405b-instruct", temperature=0.7)
index = faiss.read_index("faiss_stock.index")
corpus = pd.read_csv("stock_meta.csv")["news"].tolist()
model = SentenceTransformer("all-MiniLM-L6-v2")

@tool("Retrieve Context")
def retrieve_context(query: str, top_k: int = 4) -> Tuple[List[str], List[float]]:
    """
    Performs a semantic similarity search (vector search) against the internal
    Financial News Knowledge Base (KB) to fetch the most relevant news snippets.

    This tool is the primary method for grounding the agent's answers in validated,
    historical data before resorting to real-time external searches.

    Args:
        query (str): The specific question or search phrase to use for vector search.
                     Should be a focused query like "Tesla Q3 earnings" or "Meta AI investments."
        top_k (int, optional): The number of top-k most similar news snippets to retrieve.
                               Defaults to 4.

    Returns:
        Tuple[List[str], List[float]]: A tuple containing two lists:
            1. top_texts (List[str]): The relevant news snippets (documents) retrieved from the KB.
            2. top_distances (List[float]): The corresponding distance/score for each snippet,
                                            where a lower number indicates higher relevance/similarity.
                                            The LLM can use these distances to assess initial confidence.
    """
    if not hasattr(retrieve_context, "index") or retrieve_context.index.ntotal == 0:
        # In a real environment, you'd handle this by checking a global or passing
        # the index object. For this example, we return empty lists if the index is empty.
        # This check is essential for the "Step 1" of your workflow.
        return [], []

    query_emb = model.encode([query])
    D, I = index.search(np.array(query_emb, dtype="float32"), top_k)

    # Filter out invalid indices and gather results
    valid_indices = [i for i in I[0] if i < len(corpus)]
    top_texts = [corpus[i] for i in valid_indices]

    # Return distances corresponding to valid indices
    top_distances = [D[0][idx] for idx, i in enumerate(I[0]) if i < len(corpus)]

    return top_texts, top_distances


@tool("Find Confidence")
def check_confidence(params: dict) -> dict:
    """
    # AGENT GOAL: Evaluate whether the retrieved news context is sufficient and reliable
    # for generating a summary; decides if fallback to SerpAPI is needed.

    Uses NVIDIA NIM LLM to evaluate the sufficiency and relevance of retrieved news
    context for a given query.

    Args:
        params (dict): A dictionary containing:
            - 'query' (str): The original search query.
            - 'retrieved_texts' (list[str]): A list of news snippets or documents.

    Returns:
        dict: A dictionary containing the LLM's confidence ('high' or 'low')
              and related metadata.
    """
    query = params.get("query", "Summarize recent market news.")
    retrieved_texts = params.get("retrieved_texts", [])

    # Pre-process context
    context = (
        "\n".join(retrieved_texts) if retrieved_texts else "No relevant news found."
    )

    # Construct the prompt for the LLM
    prompt = f"""
You are a financial news analyst. Based on the following retrieved context,
decide if the information is sufficient and relevant to answer the query.

Query: {query}

Context:
{context}

Respond only with one word in lowercase:
- 'high' if the context is sufficient,
- 'low' if it is missing important recent information or context is weak.
"""

    # --- LLM API Call Configuration ---
    headers = {
        "Authorization": f"Bearer {os.getenv('NVIDIA_NIM_API_KEY')}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": NIM_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,  # Set to 0.0 for deterministic, single-word response
        "top_p": 0.5,
        "max_tokens": 10,  # Small limit for a single-word response
    }

    # --- API Call Execution and Error Handling ---
    try:
        response = requests.post(NIM_ENDPOINT, headers=headers, json=payload)
        response.raise_for_status()
        result = response.json()

        # Extract and normalize the generated content
        raw_confidence = result["choices"][0]["message"]["content"].strip().lower()

        # Normalize to strictly 'high' or 'low'
        confidence = "low" if "low" in raw_confidence else "high"

        return {
            "confidence": confidence,
            "query": query,
            "context_length": len(context),
            "model_used": NIM_MODEL,
            "generated_at": datetime.utcnow().isoformat(),
        }

    except Exception as e:
        # Fallback to 'low' confidence on API failure to trigger a search
        return {
            "confidence": "low",
            "query": query,
            "error": str(e),
            "fallback_reason": "LLM API call failed, defaulting to low confidence to trigger fallback search.",
            "generated_at": datetime.utcnow().isoformat(),
        }


@tool("Web Search")
def web_search(query: str, num_results: int = 3) -> List[str]:
    """
    AGENT TOOL: Fetches the latest news from Google News using SerpApi.

    # AGENT GOAL: Fetches latest news only when FAISS retrieval is insufficient.
    # Demonstrates dynamic tool usage to complement the knowledge base.
    # Newly fetched content is intended to be indexed to improve agent’s memory.

    Args:
        query (str): The search term for the news query.
        num_results (int): The maximum number of news results to retrieve (default: 3).

    Returns:
        List[str]: A list of formatted news strings ('Title - Snippet'),
                   or a list containing an error message if the search fails.
    """
    # 1. Get API Key securely from environment
    api_key = os.getenv("SERPAPI_API_KEY")
    if not api_key:
        return [
            "Search Error: SERPAPI_API_KEY environment variable is not set. "
            "Cannot perform live news search."
        ]

    # 2. Prepare search parameters
    params: Dict[str, Any] = {
        "engine": "google_news",
        "q": query,
        "api_key": api_key,
        "num": num_results,
        "gl": "us",  # Optional default country
        "hl": "en",  # Optional default language
    }

    try:
        # 3. Execute the search
        search = GoogleSearch(params)
        results = search.get_dict()

        news_results = results.get("news_results", [])

        # 4. Format the results
        formatted_news = [
            f"{item.get('title', 'Untitled')} - {item.get('snippet', 'No snippet available.')}"
            for item in news_results
        ]

        if not formatted_news:
            return [f"SerpApi returned no news results for the query: '{query}'."]

        return formatted_news

    except Exception as e:
        # 5. Handle any API or connection errors
        return [f"Search Error: SerpApi call failed with error: {str(e)}"]


@tool("Summarize News")
def news_summarize(text: str) -> Dict[str, Any]:
    """
    AGENT TOOL: Summarizes a block of retrieved news text into a concise,
    actionable financial report using an LLM.

    Args:
        text (str): The retrieved news context (e.g., concatenated snippets).

    Returns:
        Dict[str, Any]: A dictionary containing the summary and metadata,
                        or an error message on failure.
    """
    # Define a character limit for the input text to prevent token overflow
    MAX_TEXT_LENGTH = 10000

    # Truncate text if necessary
    context = text[:MAX_TEXT_LENGTH]

    # Construct the prompt for the LLM
    prompt = f"""
You are a highly analytical Financial News Reporter. Your task is to summarize 
the following retrieved news context.

Generate a **concise, neutral, and informative** summary in 3-5 bullet points.
The summary must focus on the key event, its financial implications, and 
any relevant market reaction.

Context to Summarize:
---
{context}
---

Summary:
"""

    # --- LLM API Call Configuration ---
    headers = {
        "Authorization": f"Bearer {os.getenv('NVIDIA_NIM_API_KEY')}",
        "Content-Type": "application/json",
    }

    # Adjust max_tokens for a full summary (e.g., 512 for a few paragraphs)
    payload = {
        "model": NIM_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3,  # Allow some creativity for good flow, but keep it factual
        "top_p": 0.9,
        "max_tokens": 512,  # Sufficient limit for a detailed summary
    }

    # --- API Call Execution and Error Handling ---
    try:
        response = requests.post(NIM_ENDPOINT, headers=headers, json=payload)
        response.raise_for_status()
        result = response.json()

        # Extract the generated content
        summary = result["choices"][0]["message"]["content"].strip()

        return {
            "summary": summary,
            "model_used": NIM_MODEL,
            "generated_at": datetime.utcnow().isoformat(),
        }

    except Exception as e:
        # Return an empty summary string on failure
        return {
            "summary": "",
            "error": str(e),
            "fallback_reason": "LLM API call failed; unable to generate summary.",
            "generated_at": datetime.utcnow().isoformat(),
        }


tools = [retrieve_context,  # Step 1: Retrieve context from DB
        check_confidence,  # Step 2: Find whether this context is sufficient
        web_search,  # Step 3: If low confidence → Fetch from SerpApi
        news_summarize] # Step 3: Summarize fresh news]

news_agent = Agent(
    role="Financial News Strategist",
    backstory="""
    A meticulous and resourceful news and document intelligence analyst. Your primary directive is to prioritize 
    internal, validated knowledge bases (like official SEC filings or proprietary research) to ensure accuracy. 
    You only use external web searches as a last resort when internal data is too old or incomplete.
    """,
    goal="""
    Execute a strategic, multi-step search for financial context and news for a given query.
    1. Always attempt to retrieve and validate context from the internal database first.
    2. If confidence in internal data is low, immediately use external web search (SerpApi) to find fresh news.
    3. Index any new, high-quality external news found into the database.
    4. Return a definitive, summarized result based on the highest confidence source.
    """,
    instructions="""
    - **Step 1: Internal Search & Confidence Check** - Use 'retrieve_context' and then 'check_confidence' on the retrieved context.
    - **Step 2: Low Confidence Action** - If 'check_confidence' returns 'low', use 'web_search' for fresh, real-time news. Summarize the fresh news using 'nim_summarize' and then update the internal database using 'add_to_faiss'.
    - **Step 3: High Confidence Action** - If confidence is 'high', use the retrieved internal context directly.
    - **Output**: The final output must clearly indicate the source ('internal_db' or 'web') and provide a final summary of the findings.
    """,
    tools=tools,
    llm=llm,
)
# Placeholder for a user-provided query, incorporating complex elements:
user_query = "What about Meta AI investments and ad revenue growth? Also, check for the latest news on the Tesla chip supply shortage."

financial_news_task = Task(
    description=f"""
    Execute a comprehensive news and document intelligence scan based on the user's query: '{user_query}'

    The agent must identify all distinct topics and tickers in the query (e.g., META, TSLA, AI, revenue, supply shortage) and address each one.

    Examples of input queries the agent must be able to handle include:
    - "Tesla Q3 earnings report"
    - "Apple chip supply shortage"
    - "What are Microsoft AI initiatives?"
    - "What about Meta AI investments and ad revenue growth?"

    The critical steps are:
    1. **Knowledge Base (KB) Scan**: For each identified company, search the internal KB for all official documents related to the specific topics (e.g., META earnings call for 'ad revenue').
    2. **Conditional External News Check**: Use 'web_search' only when the KB lacks recent or specific information on a topic (e.g., 'chip supply shortage' might be a real-time news item).
    3. **Synthesis**: Combine all validated insights into a single, comprehensive report addressing all parts of the original query.
    """,
    agent=news_agent,  # Assign the specialized agent
    expected_output=f"""
    A single, structured JSON object that fully addresses every component of the user's query, ensuring each piece of information is sourced from either the Knowledge Base (KB) or External News (Web Search).

    {{
        "analysis_timestamp": "ISO-8601",
        "original_query": "{user_query}",
        "reports": [
            {{
                "entity": "META",
                "topic": "AI Investments & Ad Revenue Growth",
                "source_priority": "KB",
                "summary": "Key metrics and official commentary from the last earnings report on these topics.",
                "confidence_score": 0.95
            }},
            {{
                "entity": "TSLA",
                "topic": "Chip Supply Shortage",
                "source_priority": "External News",
                "summary": "Latest news updates, market impact, and company statements regarding the shortage.",
                "confidence_score": 0.88
            }},
            // ... Add reports for any other entities/topics identified.
        ]
    }}
    """,
)
# Create the Crew
news_crew = Crew(
    agents=[news_agent],
    tasks=[financial_news_task],
    verbose=True,  # Recommended to see the execution steps
)

# Execute the crew
result = news_crew.kickoff()
print(result)
