# Semantic Caching: Two-Tier LLM Cache & Inference Engine

> **Cut LLM latency from 500ms down to 0.01ms and slash token bills by 60–80% without sacrificing response quality.**

---

## The Problem Every LLM Developer Hits

If you have ever shipped an LLM app to production, you know the sinking feeling of watching your API bill climb while your users complain about 500ms to 2-second response delays.

When we looked at real-world production query traffic, an obvious pattern emerged:
- **A huge portion of user queries are either identical or semantic variations of questions already asked.**
- Questions like:
  - *"What is the boiling point of water at sea level?"*
  - *"At what temperature does water boil at sea level?"*
  - *"How hot does water need to be to boil?"*

In a standard setup, every single one of those questions triggers a brand-new cloud API call to an LLM. You pay for the prompt tokens, you pay for the generation tokens, and your user sits waiting while the model generates the exact same answer it generated thirty seconds ago for someone else.

Traditional web caches (like Redis exact-key lookups) fail completely here because string hashing is brittle: one typo, a slight paraphrase, or a flipped word order results in a 100% cache miss.

**This project solves that problem.**

We built an inference server architecture across two core components:
- **`server.py`**: The high-performance FastAPI backend with a Two-Tier Cache (Tier 1 Exact Match + Tier 2 Semantic Cosine Similarity) backed by Groq LPU inference.
- **`index.html`**: A glassmorphism interactive playground and real-time cache inspector dashboard.

---

## Architecture & How It Works

```mermaid
flowchart TD
    A["User Prompt (via index.html or REST API)"] --> B["server.py: Tier 1 Exact Match Hash Lookup"]
    B -- "Hit (Identical String)" --> C["Return Cached Answer (~0.01ms)"]
    B -- "Miss" --> D["Compute Normalized Sentence Embedding (all-MiniLM-L6-v2)"]
    D --> E["server.py: Tier 2 Cosine Similarity Scan vs. Cached Vectors"]
    E -- "Cosine Similarity >= Threshold (e.g. 0.90)" --> F["Return Semantic Match Answer (~10-25ms)"]
    E -- "Below Threshold / Cold Cache" --> G["Groq LPU LLM Inference (qwen3.8-27b)"]
    G --> H["Store Response & Embedding in Cache"]
    H --> I["Return Fresh Generated Answer (~200-400ms)"]
```

### 1. Tier 1: Exact Match Lookup (`server.py`)
- Hashed against `(clean_prompt, max_new_tokens)`.
- Checks an in-memory dictionary with active TTL verification.
- **Latency**: `< 0.05 ms` (instantaneous hash lookup).
- **Cost**: **$0.00**.

### 2. Tier 2: Semantic Similarity Match (`server.py`)
- If Tier 1 misses, `server.py` encodes the prompt into a 384-dimensional dense vector using `sentence-transformers/all-MiniLM-L6-v2`.
- Mean pooling is applied with an attention mask, followed by $L_2$ vector normalization.
- Calculates the dot product (cosine similarity) against stored candidate vectors with matching token budgets.
- If the highest score exceeds the configurable similarity threshold (default `0.90`), the cached answer is returned immediately.
- **Latency**: `~10 ms – 30 ms` (CPU-bound local vector dot product).
- **Cost**: **$0.00**.

### 3. Graceful Miss & Auto-Populate
- If neither tier matches, the prompt is dispatched directly to Groq's high-speed completion API (`qwen/qwen3.8-27b`).
- Upon receiving the completion, `server.py` saves the prompt text, model output, token telemetry, and pre-computed embedding into the cache.
- Future variations of this prompt instantly hit Tier 1 or Tier 2.

---

## The Process: How We Built It (From Naive Script to Production Server)

Building this wasn't an overnight "download a library and call it a day" project. We went through distinct iterative phases, running into real engineering bottlenecks and learning what works and what doesn't.

```
[Stage A: Raw Prototype]  ──>  [Stage B: Telemetry & Metrics]
                                        │
                                        ▼
[Stage D: Exact Caching]  <──  [Stage C: The Batching Experiment]
         │
         ▼
[Stage E: Two-Tier Semantic Architecture (server.py + index.html)]
```

### Stage A: The Raw Prototype
We started simple: a minimal FastAPI endpoint in `server.py` wrapping direct calls to Groq API.
- *What worked*: Groq was fast (~300–400ms).
- *The bottleneck*: Testing the same query or slightly different prompts repeatedly drained API rate limits quickly and added unnecessary network round-trip delay.

### Stage B: Observability & Telemetry
Before optimizing, we needed real data. We added structured JSONL logging in `server.py`:
- Prompt character length vs. token counts (prompt tokens, generated tokens, total tokens).
- Precise latency breakdown: request arrival timestamp, processing time, and total latency.
- Unique request IDs (`req-xxxxxxxx`) for end-to-end trace auditing.

### Stage C: The Dynamic Batching Experiment
Next, we tested dynamic request batching:
- We implemented an `asyncio.Queue` and a background worker collecting requests over a time window (`BATCH_TIMEOUT = 200ms`, `MAX_BATCH_SIZE = 4`).
- *The revelation*: While batching is helpful for local GPU matrix multiplications, in an interactive user-facing API using cloud endpoints, **batch timeout queues introduce artificial delay for single incoming requests**. A user with an urgent question had to wait up to 200ms just to see if another user would show up.
- *The decision*: We stripped out the batch queuing mechanism completely. Individual misses fly straight to the model with zero wait, while the caching tiers handle throughput and cost reduction.

### Stage D: Exact-Match In-Memory Cache
We introduced an exact string cache with automatic TTL expiration into `server.py`:
- Identical queries dropped from **400ms to 0.05ms** — an 8,000x speedup!
- *The catch*: Real users rarely type identical strings. A user typing `"What is sun's size?"` got zero benefit if the cache held `"How big is the sun?"`. We were missing over 70% of potential cache hits due to minor lexical differences.

### Stage E: The Two-Tier Semantic Architecture (Current)
To solve the paraphrase problem, we engineered Tier 2:
- We embedded a lightweight HuggingFace sentence transformer (`all-MiniLM-L6-v2`, ~90MB) directly inside `server.py`.
- We tuned the cosine similarity threshold to the sweet-spot range (**0.88 – 0.90**) where semantic hits are reliable without risking false positives on opposite negations or entity swaps.
- Finally, we built `index.html` — a responsive glassmorphism web playground and live cache dashboard connected to `server.py`'s API endpoints.

---

## Features

- **Two-Tier Cache Hierarchy (`server.py`)**:
  - **Tier 1 (Exact Match)**: Instant $O(1)$ hash map lookup.
  - **Tier 2 (Semantic Cosine Match)**: Embedding-based similarity matching using sentence transformers.
- **Configurable Similarity Threshold**: Query parameter control from `0.70` to `0.99` (recommended: `0.88` – `0.92`).
- **Automatic TTL & Lazy Eviction**: Expired entries are safely purged on lookup or background refresh without background thread overhead.
- **Granular & Global Invalidation**: Evict an individual query or flush the entire cache via `/clear_cache`.
- **Interactive Web Playground & Dashboard (`index.html`)**:
  - Live prompt runner with instant paraphrase test chips (*"Water Boiling Point"*, *"Paraphrase (Tier 2 Test)"*, *"Miss Test"*).
  - Real-time telemetry cards (Average Latency, Tokens/Sec, Cache Hit Rate).
  - Active cache store table with live age counter and one-click item eviction.
  - Deep Request Inspector Modal displaying raw JSON telemetry and exportable cURL commands.
- **Queue-Free Model Execution**: Zero artificial queuing delays on cache misses.

---

## Real-Life Applications & Why It Matters

### 1. High-Volume Customer Support & FAQ Bots
In customer service, 70–80% of incoming tickets revolve around the same 50 topics (*"How do I track my order?"*, *"Where is my package?"*, *"Status of my shipment"*).
- **Without this system**: Every variation hits the cloud LLM, creating huge monthly bills and unpredictable latency spikes.
- **With Two-Tier Cache**: The first question populates the cache. The next 5,000 customers asking the same question in different phrasing receive their answers in **under 20 milliseconds**, costing **$0**.

### 2. E-Commerce Search & Product Q&A
Shoppers frequently ask repetitive questions on product pages:
- *"Does this fit a 2024 Honda Civic?"*
- *"Is this compatible with Honda Civic 2024 model?"*
Semantic caching allows instant retrieval without recurring API charges.

### 3. Internal Company Knowledge Bases
Internal engineering help desks repeatedly answer setup questions (*"How do I set up VPN?"*, *"Steps to configure company VPN"*). Semantic caching eliminates redundant queries across team members.

### 4. Protecting Against Rate Limits & Traffic Spikes
Third-party LLM providers enforce tight Requests-Per-Minute (RPM) and Tokens-Per-Minute (TPM) limits. By serving 60–80% of queries directly from local memory:
- Your application can handle 5x to 10x higher user concurrency.
- You avoid `429 Rate Limit Exceeded` errors during traffic spikes.

---

## Performance Benchmarks

Measured performance on `server.py`:

| Metric | Cold Baseline (Direct Model Inference) | Two-Tier Cache Active | Improvement |
| :--- | :--- | :--- | :--- |
| **P50 Latency** | **159.06 ms** | **0.01 ms** | **~15,000x faster** |
| **P90 Latency** | 166.36 ms | 31.19 ms (semantic hit) | **5.3x faster** |
| **P99 Latency** | 186.71 ms | 35.42 ms | **5.2x faster** |
| **Throughput (req/s)** | ~60 req/s | **120+ req/s** | **2x capacity** |
| **Token Cost** | 100% billed | **33.3% billed** (66.7% hit rate) | **~67% cost reduction** |

---

## Getting Started

### 1. Prerequisites
- Python 3.10+
- A free [Groq API Key](https://console.groq.com)
- Git

### 2. Clone the Repository
Clone the **Semantic Caching** repository and enter the directory:
```bash
git clone https://github.com/<your-username>/Semantic-Caching.git
cd Semantic-Caching
```

### 3. Create & Activate a Virtual Environment
Setting up an isolated virtual environment is recommended:
```bash
# Create virtual environment
python -m venv venv

# Activate on Windows (Command Prompt):
venv\Scripts\activate.bat

# Activate on Windows (PowerShell):
venv\Scripts\Activate.ps1

# Activate on Linux / macOS:
source venv/bin/activate
```

### 4. Install Dependencies
Install the required packages using [`requirements.txt`](file:///requirements.txt):
```bash
pip install -r requirements.txt
```

> [!NOTE]
> `server.py` natively loads the `sentence-transformers/all-MiniLM-L6-v2` embedding model using Hugging Face's official `transformers` and `torch` packages with attention-masked mean pooling and $L_2$ normalization. The standalone `sentence-transformers` library is not required.
>
> If you are on a CPU-only system and want a lightweight PyTorch install:
> ```bash
> pip install torch --index-url https://download.pytorch.org/whl/cpu
> pip install -r requirements.txt
> ```

### 5. Configuration
Copy the `.env.example` file to create your `.env`:
```bash
# Windows (CMD):
copy .env.example .env

# Windows (PowerShell):
Copy-Item .env.example .env

# Linux / macOS:
cp .env.example .env
```
Open `.env` and set your Groq API credentials:
```ini
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL_NAME=qwen/qwen3.8-27b
```

### 6. Start the Inference Server
Run `server.py`:
```bash
python server.py
```
The server will initialize the local embedding model and start on:
`http://127.0.0.1:8004`

---

## Web Playground & Dashboard (`index.html`)

Open your browser and navigate to:
```
http://127.0.0.1:8004/playground
```

`server.py` serves `index.html` at both `/` and `/playground`.

### What You Can Do in the UI:
1. **Interactive Prompt Runner**:
   - Click the test chips to see the cache in action.
   - Watch the status badge transition from **Cache Miss** $\rightarrow$ **Tier 1 Exact Match (100%)** $\rightarrow$ **Tier 2 Semantic Match (Cosine Score)**.
2. **Dynamic Sliders**:
   - Adjust **Max Tokens** and the **Similarity Threshold** on the fly.
3. **Telemetry & Live Cache Inspector**:
   - Switch to the **Cache Inspector & Dashboard** tab to view all active cache entries, their age, and evict individual prompts with a single click.
4. **Request Inspector**:
   - Click any request tag (e.g. `req-a1b2c3d4`) to open the modal inspection window with full prompt, output, latency breakdown, and a copyable cURL command.

---

## API Reference (`server.py`)

### `POST /generate`
Generate a completion with two-tier cache lookup.

**Query Parameters**:
- `enable_cache` (bool, default: `true`): Toggle caching overall.
- `enable_semantic_cache` (bool, default: `true`): Toggle Tier 2 semantic matching.
- `similarity_threshold` (float, default: `0.90`): Cosine cutoff for semantic match (range: 0.50 – 1.00).
- `cache_ttl` (float, default: `3600.0`): Time-to-live in seconds.

**Request Body**:
```json
{
  "prompt": "What temperature does water boil at?",
  "max_new_tokens": 20
}
```

**Response Example (Tier 2 Semantic Hit)**:
```json
{
  "request_id": "req-8f92a11b",
  "prompt": "What temperature does water boil at?",
  "generated_text": "At standard atmospheric pressure, water boils at 100 degrees Celsius (212 degrees Fahrenheit).",
  "processing_time_ms": 0.0,
  "total_latency_ms": 14.82,
  "cache_hit": true,
  "match_type": "semantic",
  "similarity_score": 0.9412,
  "matched_prompt": "What is the boiling point of water at sea level?",
  "prompt_tokens": 12,
  "generated_tokens": 20,
  "total_model_tokens": 0
}
```

### `POST /clear_cache`
Invalidate cache entries in `server.py`.
- To clear a specific prompt: `POST /clear_cache?prompt=Your+prompt+here`
- To clear the entire cache: `POST /clear_cache`

### `GET /api/status`
Returns active model info, cache size, and TTL configuration.

### `GET /api/cache_entries`
Returns an array of all active entries currently in memory with age in seconds.

### `GET /api/logs`
Fetches recent telemetry records from `telemetry_metrics.jsonl`.

### `GET /playground` or `GET /`
Serves the `index.html` frontend interface.

---

## Guardrails & Gotchas Learned

1. **Why `max_new_tokens` is part of the cache key in `server.py`**:
   A prompt asking for `max_new_tokens = 10` produces a truncated sentence, while `max_new_tokens = 100` produces a complete paragraph. Matching across different token limits would lead to incomplete answers.
2. **Why threshold tuning matters (Adversarial Negations)**:
   Sentence embeddings excel at identifying synonyms, but can score high on negation opposites (e.g. *"Why is Python fast?"* vs *"Why is Python NOT fast?"*). Keeping the threshold at `0.88 – 0.92` ensures true paraphrases match while blocking negations and entity swaps.
3. **Lazy TTL Eviction**:
   Instead of running resource-heavy periodic cleanup threads, expirations in `server.py` are checked on-demand during lookups and status queries, keeping memory usage clean and CPU usage zero when idle.

---

## AWS Lambda Serverless Architecture (New)

The inference server and Two-Tier Semantic Cache can run fully serverless on **AWS Lambda** in your project's selected Region (`ap-southeast-2`):

```mermaid
flowchart TD
    User["Client / Web Browser"] -->|HTTPS| LUrl["AWS Lambda Function URL"]
    LUrl --> LHandler["Lambda Handler (Mangum + FastAPI)"]
    
    subgraph LambdaContainer ["AWS Lambda Runtime (ap-southeast-2)"]
        LHandler --> L1["L1 Cache: In-Memory Warm Dictionary (<0.1ms)"]
        LHandler --> EmbRouter["Embedding Engine (Bedrock Titan v2 / Local)"]
    end
    
    subgraph ManagedAWS ["AWS Managed Services (ap-southeast-2)"]
        L1 -- "Miss" --> DDB["Amazon DynamoDB: llm-semantic-cache (TTL enabled)"]
        EmbRouter -- "Compute Embedding" --> Bedrock["Amazon Bedrock: amazon.titan-embed-text-v2:0"]
        LHandler -- "Structured Telemetry" --> DDBTel["Amazon DynamoDB: llm-telemetry-logs"]
        LHandler -- "System Traces" --> CW["Amazon CloudWatch Logs"]
    end
    
    LHandler -- "Cache Miss Fallback" --> Groq["Groq LPU API (qwen/qwen3.8-27b)"]
```

### Key Serverless Refinements:
1. **Hybrid Two-Tier Cache (L1 Warm Memory + L2 DynamoDB)**:
   - **L1 In-Memory**: Instant sub-millisecond lookups during warm Lambda execution.
   - **L2 Amazon DynamoDB (`llm-semantic-cache`)**: Cross-instance shared persistence and candidate vector matching across cold starts and auto-scaling instances. Automatically purges expired entries via native DynamoDB TTL (`ttl_timestamp`).
2. **Lean Serverless Embeddings via Amazon Bedrock**:
   - Uses `amazon.titan-embed-text-v2:0` directly in `ap-southeast-2`.
   - Eliminates the ~1.5 GB PyTorch / Transformers dependencies in the Lambda container, shrinking the deployment package down to **5.7 MB** with sub-second cold starts!
   - Full backward compatibility: Automatically falls back to HuggingFace or N-gram vectors when running locally without AWS credentials.
3. **Persistent Telemetry Traces (`llm-telemetry-logs`)**:
   - Overcomes Lambda's ephemeral filesystem by persisting structured request telemetry directly into DynamoDB and CloudWatch Logs.
4. **Zero-Cost Public HTTPS Endpoint**:
   - Configured with an **AWS Lambda Function URL** (AuthType: `NONE` with CORS enabled), serving both the REST APIs (`/generate`, `/clear_cache`, `/api/*`) and the interactive playground (`/playground`) with zero API Gateway hourly fees.

### Automated Deployment to AWS Lambda

Deploy the entire stack with a single command:
```bash
python deploy_lambda.py
```
This automatically:
- Provisions DynamoDB tables (`llm-semantic-cache` and `llm-telemetry-logs`) with On-Demand billing.
- Sets up the IAM execution role (`LLMCacheLambdaExecutionRole`).
- Bundles the 5.7 MB deployment zip package (`lambda_deployment.zip`).
- Creates or updates the Lambda function (`llm-semantic-cache-service`).
- Configures the public HTTPS Function URL.

### Testing Locally vs. Live Lambda

- **Run tests against local server**:
  ```bash
  python test.py
  ```
- **Run tests against live AWS Lambda Function URL**:
  ```bash
  python test.py https://<your-lambda-url>.lambda-url.ap-southeast-2.on.aws
  ```

---
