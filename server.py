import asyncio
from contextlib import asynccontextmanager
from dataclasses import dataclass
import json
import logging
import os
import signal
import time
from typing import Optional
import uuid

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
import httpx
from mangum import Mangum
from pydantic import BaseModel, Field

from aws_storage import (
    AWS_REGION,
    DynamoDBCacheStore,
    DynamoDBTelemetryStore,
    EmbeddingManager,
    compute_cosine_similarity,
)

# Load environment variables from .env file
load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("inference_server")

# Configuration
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "").strip()
GROQ_MODEL_NAME = os.environ.get("GROQ_MODEL_NAME", "qwen/qwen3.8-27b")
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

LOG_FILE_PATH = os.environ.get("LOG_FILE_PATH", "telemetry_metrics.jsonl")
DEFAULT_CACHE_TTL_SECONDS = float(os.environ.get("DEFAULT_CACHE_TTL_SECONDS", "3600.0"))
AUTO_SHUTDOWN_SECONDS = float(os.environ.get("AUTO_SHUTDOWN_SECONDS", "0"))
INDEX_HTML_PATH = "index.html" if os.path.exists("index.html") else os.path.join("static", "index.html")

# AWS and Local Providers
embedding_manager = EmbeddingManager()
dynamodb_cache_store = DynamoDBCacheStore()
dynamodb_telemetry_store = DynamoDBTelemetryStore()


def compute_embedding(prompt: str) -> Optional[list[float]]:
    """Compute normalized sentence embedding using the active provider (Bedrock Titan or Local)."""
    return embedding_manager.compute_embedding(prompt)


@dataclass(slots=True)
class CacheEntry:
    prompt: str
    max_new_tokens: int
    response_text: str
    embedding: Optional[list[float]]
    timestamp: float
    prompt_tokens: int = 0
    generated_tokens: int = 0
    total_model_tokens: int = 0

    def is_expired(self, max_age_seconds: float) -> bool:
        return (time.time() - self.timestamp) > max_age_seconds


class TwoTierCache:
    """Two-tier cache combining exact string lookup (Tier 1) and semantic similarity (Tier 2).

    Supports:
    - L1 Warm In-Memory Cache (fast, sub-millisecond)
    - L2 Serverless Persistent Cache (Amazon DynamoDB with TTL)
    """

    def __init__(
        self,
        default_ttl_seconds: float = DEFAULT_CACHE_TTL_SECONDS,
        dynamodb_store: Optional[DynamoDBCacheStore] = None
    ):
        self.default_ttl_seconds = default_ttl_seconds
        self.entries: dict[tuple[str, int], CacheEntry] = {}
        self.dynamodb_store = dynamodb_store or dynamodb_cache_store

    def _evict_expired(self, max_age_seconds: Optional[float] = None) -> None:
        ttl = max_age_seconds if max_age_seconds is not None else self.default_ttl_seconds
        now = time.time()
        expired = [k for k, v in self.entries.items() if (now - v.timestamp) > ttl]
        for k in expired:
            del self.entries[k]

    def lookup_exact(
        self,
        prompt: str,
        max_new_tokens: int,
        max_age_seconds: Optional[float] = None
    ) -> Optional[CacheEntry]:
        self._evict_expired(max_age_seconds)
        clean = prompt.strip()
        key = (clean, max_new_tokens)

        # 1. Check L1 In-Memory Cache
        if key in self.entries:
            return self.entries[key]

        # 2. Check L2 DynamoDB Persistent Cache
        if self.dynamodb_store and self.dynamodb_store.is_available:
            ttl = max_age_seconds if max_age_seconds is not None else self.default_ttl_seconds
            item = self.dynamodb_store.get_exact(clean, max_new_tokens, ttl)
            if item:
                entry = CacheEntry(
                    prompt=item.get("prompt", clean),
                    max_new_tokens=item.get("max_new_tokens", max_new_tokens),
                    response_text=item.get("response_text", ""),
                    embedding=item.get("embedding"),
                    timestamp=item.get("timestamp", time.time()),
                    prompt_tokens=item.get("prompt_tokens", 0),
                    generated_tokens=item.get("generated_tokens", 0),
                    total_model_tokens=item.get("total_model_tokens", 0)
                )
                # Populate warm L1 cache
                self.entries[key] = entry
                return entry

        return None

    def lookup_semantic(
        self,
        prompt_embedding: list[float],
        max_new_tokens: int,
        similarity_threshold: float,
        max_age_seconds: Optional[float] = None
    ) -> tuple[Optional[CacheEntry], Optional[float]]:
        self._evict_expired(max_age_seconds)

        # 1. Search L1 In-Memory Candidates
        candidates = [
            entry for (_, tok), entry in self.entries.items()
            if tok == max_new_tokens and entry.embedding is not None
        ]

        best_entry: Optional[CacheEntry] = None
        best_sim = -1.0

        for entry in candidates:
            try:
                sim = compute_cosine_similarity(prompt_embedding, entry.embedding)
                if sim > best_sim:
                    best_sim = sim
                    best_entry = entry
            except Exception as e:
                logger.warning("Error comparing embedding against cached prompt '%s': %s", entry.prompt, e)

        if best_entry is not None and best_sim >= similarity_threshold:
            return best_entry, round(best_sim, 4)

        # 2. Search L2 DynamoDB Candidates (Cross-instance shared cache)
        if self.dynamodb_store and self.dynamodb_store.is_available:
            ttl = max_age_seconds if max_age_seconds is not None else self.default_ttl_seconds
            db_candidates = self.dynamodb_store.get_candidates(max_new_tokens, ttl)
            for item in db_candidates:
                emb = item.get("embedding")
                if not emb:
                    continue
                try:
                    sim = compute_cosine_similarity(prompt_embedding, emb)
                    if sim > best_sim:
                        best_sim = sim
                        best_entry = CacheEntry(
                            prompt=item.get("prompt", ""),
                            max_new_tokens=item.get("max_new_tokens", max_new_tokens),
                            response_text=item.get("response_text", ""),
                            embedding=emb,
                            timestamp=item.get("timestamp", time.time()),
                            prompt_tokens=item.get("prompt_tokens", 0),
                            generated_tokens=item.get("generated_tokens", 0),
                            total_model_tokens=item.get("total_model_tokens", 0)
                        )
                        # Warm L1 cache with this candidate
                        self.entries[(best_entry.prompt, max_new_tokens)] = best_entry
                except Exception as e:
                    logger.warning("Error comparing DynamoDB candidate: %s", e)

        if best_entry is not None and best_sim >= similarity_threshold:
            return best_entry, round(best_sim, 4)

        return None, round(best_sim, 4) if best_entry is not None else None

    def insert(
        self,
        prompt: str,
        max_new_tokens: int,
        response_text: str,
        embedding: Optional[list[float]] = None,
        prompt_tokens: int = 0,
        generated_tokens: int = 0,
        total_model_tokens: int = 0
    ) -> CacheEntry:
        clean = prompt.strip()
        if embedding is None:
            embedding = compute_embedding(clean)

        entry = CacheEntry(
            prompt=clean,
            max_new_tokens=max_new_tokens,
            response_text=response_text,
            embedding=embedding,
            timestamp=time.time(),
            prompt_tokens=prompt_tokens,
            generated_tokens=generated_tokens,
            total_model_tokens=total_model_tokens
        )
        # Populate L1
        self.entries[(clean, max_new_tokens)] = entry

        # Persist to L2 DynamoDB
        if self.dynamodb_store and self.dynamodb_store.is_available:
            self.dynamodb_store.put_entry(
                prompt=clean,
                max_new_tokens=max_new_tokens,
                response_text=response_text,
                embedding=embedding,
                prompt_tokens=prompt_tokens,
                generated_tokens=generated_tokens,
                total_model_tokens=total_model_tokens,
                ttl_seconds=self.default_ttl_seconds
            )

        return entry

    def clear(self, prompt: Optional[str] = None) -> int:
        count = 0
        if prompt:
            clean = prompt.strip()
            keys = [k for k in self.entries if k[0] == clean]
            for k in keys:
                del self.entries[k]
            count = len(keys)
        else:
            count = len(self.entries)
            self.entries.clear()

        # Clear in DynamoDB
        if self.dynamodb_store and self.dynamodb_store.is_available:
            db_cleared = self.dynamodb_store.clear(prompt)
            count = max(count, db_cleared)

        return count


# App State
two_tier_cache = TwoTierCache(dynamodb_store=dynamodb_cache_store)
http_client: Optional[httpx.AsyncClient] = None


def append_telemetry_log(record: dict) -> None:
    # 1. DynamoDB Telemetry Table
    if dynamodb_telemetry_store.is_available:
        dynamodb_telemetry_store.append_log(record)

    # 2. Local JSONL File (Fallback & local dev)
    try:
        with open(LOG_FILE_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
    except Exception as e:
        logger.debug("Could not write to local log file: %s", e)


async def async_process_prompt(prompt: str, max_new_tokens: int) -> dict:
    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY is not configured. Please set GROQ_API_KEY in your .env file or environment variables.")
    start_time = time.time()
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    client = http_client or httpx.AsyncClient(timeout=30.0)
    resp = await client.post(
        GROQ_API_URL,
        headers=headers,
        json={
            "model": GROQ_MODEL_NAME,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_new_tokens
        }
    )
    if resp.status_code != 200:
        raise RuntimeError(f"Groq API error ({resp.status_code}): {resp.text}")

    data = resp.json()
    gen_text = data["choices"][0]["message"]["content"] or ""
    usage = data.get("usage", {})
    p_tok = usage.get("prompt_tokens", len(prompt.split()))
    g_tok = usage.get("completion_tokens", len(gen_text.split()))
    t_tok = usage.get("total_tokens", p_tok + g_tok)
    proc_time_sec = time.time() - start_time

    return {
        "generated_text": gen_text,
        "processing_time_sec": proc_time_sec,
        "prompt_tokens": p_tok,
        "generated_tokens": g_tok,
        "total_model_tokens": t_tok
    }


@asynccontextmanager
async def lifespan(app: FastAPI):
    global http_client
    http_client = httpx.AsyncClient(timeout=30.0)
    shutdown_task = None

    if AUTO_SHUTDOWN_SECONDS > 0:
        async def shutdown_after_timeout():
            await asyncio.sleep(AUTO_SHUTDOWN_SECONDS)
            logger.info("Automatic shutdown timer reached; stopping server.")
            os.kill(os.getpid(), signal.SIGINT)

        shutdown_task = asyncio.create_task(shutdown_after_timeout())

    try:
        yield
    finally:
        if shutdown_task is not None:
            shutdown_task.cancel()
            await asyncio.gather(shutdown_task, return_exceptions=True)
        if http_client is not None:
            await http_client.aclose()


app = FastAPI(
    title="LLM Inference Server - Two-Tier Cache on AWS Lambda",
    description="Inference server with Tier 1 exact match cache and Tier 2 semantic embedding cache on AWS Lambda & DynamoDB.",
    lifespan=lifespan
)


# Request / Response Schemas
class GenerateRequest(BaseModel):
    prompt: str = Field(..., description="Input prompt text", min_length=1)
    max_new_tokens: int = Field(default=20, ge=1, le=100)


class GenerateResponse(BaseModel):
    request_id: str
    prompt: str
    generated_text: str
    processing_time_ms: float
    total_latency_ms: float
    cache_hit: bool
    match_type: str = Field(default="none", description="Match type: 'exact', 'semantic', or 'none'")
    similarity_score: Optional[float] = Field(default=None, description="Similarity score for semantic match")
    matched_prompt: Optional[str] = Field(default=None, description="Original prompt matched against")
    prompt_tokens: int
    generated_tokens: int
    total_model_tokens: int


def _record_and_build_response(
    request_id: str,
    prompt: str,
    generated_text: str,
    arrival_time: float,
    proc_time_ms: float,
    cache_hit: bool,
    match_type: str,
    similarity_score: Optional[float],
    matched_prompt: Optional[str],
    prompt_tokens: int,
    generated_tokens: int,
    total_model_tokens: int,
    enable_cache: bool,
    enable_semantic_cache: bool,
    similarity_threshold: float
) -> GenerateResponse:
    total_latency_ms = (time.time() - arrival_time) * 1000.0

    response = GenerateResponse(
        request_id=request_id,
        prompt=prompt,
        generated_text=generated_text,
        processing_time_ms=round(proc_time_ms, 2),
        total_latency_ms=round(total_latency_ms, 2),
        cache_hit=cache_hit,
        match_type=match_type,
        similarity_score=similarity_score,
        matched_prompt=matched_prompt,
        prompt_tokens=prompt_tokens,
        generated_tokens=generated_tokens,
        total_model_tokens=total_model_tokens
    )

    append_telemetry_log({
        "request_id": request_id,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(arrival_time)),
        "prompt": prompt,
        "generated_text": generated_text,
        "prompt_char_len": len(prompt.strip()),
        "max_new_tokens": total_model_tokens,
        "processing_time_ms": round(proc_time_ms, 2),
        "total_latency_ms": round(total_latency_ms, 2),
        "cache_hit": cache_hit,
        "match_type": match_type,
        "similarity_score": similarity_score,
        "matched_prompt": matched_prompt,
        "prompt_tokens": prompt_tokens,
        "generated_tokens": generated_tokens,
        "total_model_tokens": total_model_tokens,
        "cache_enabled": enable_cache,
        "semantic_cache_enabled": enable_semantic_cache,
        "similarity_threshold": similarity_threshold
    })

    return response


# Endpoints
@app.get("/")
@app.get("/playground")
def playground():
    if os.path.exists(INDEX_HTML_PATH):
        return FileResponse(INDEX_HTML_PATH)
    return {"message": "Semantic Cache LLM Inference Server (AWS Lambda)"}


@app.get("/api/status")
def status_api():
    total_entries = len(two_tier_cache.entries)
    if dynamodb_cache_store.is_available:
        try:
            db_entries = len(dynamodb_cache_store.list_entries(limit=100))
            total_entries = max(total_entries, db_entries)
        except Exception:
            pass

    return {
        "status": "ok",
        "stage": "Two-Tier Cache Architecture (AWS Lambda + DynamoDB)",
        "aws_region": AWS_REGION,
        "embedding_provider": embedding_manager.provider_name,
        "dynamodb_cache_available": dynamodb_cache_store.is_available,
        "dynamodb_telemetry_available": dynamodb_telemetry_store.is_available,
        "generative_model": GROQ_MODEL_NAME,
        "log_file": LOG_FILE_PATH,
        "cache_entries": total_entries,
        "cache_ttl_seconds": two_tier_cache.default_ttl_seconds
    }


@app.get("/api/cache_entries")
def get_cache_entries():
    # 1. DynamoDB Entries (Cross-instance)
    if dynamodb_cache_store.is_available:
        db_items = dynamodb_cache_store.list_entries(limit=100)
        if db_items:
            return db_items

    # 2. Local Fallback
    two_tier_cache._evict_expired()
    now = time.time()
    return [
        {
            "prompt": entry.prompt,
            "max_new_tokens": entry.max_new_tokens,
            "response_preview": entry.response_text[:80] + ("..." if len(entry.response_text) > 80 else ""),
            "timestamp": entry.timestamp,
            "age_seconds": round(now - entry.timestamp, 1),
            "has_embedding": entry.embedding is not None
        }
        for entry in two_tier_cache.entries.values()
    ]


@app.get("/api/logs")
def get_logs(limit: int = Query(default=50, ge=1, le=500)):
    # 1. DynamoDB Logs
    if dynamodb_telemetry_store.is_available:
        db_logs = dynamodb_telemetry_store.get_logs(limit=limit)
        if db_logs:
            return db_logs

    # 2. Local File Fallback
    if not os.path.exists(LOG_FILE_PATH):
        return []
    records = []
    with open(LOG_FILE_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line_str = line.strip()
            if line_str:
                try:
                    records.append(json.loads(line_str))
                except Exception:
                    pass
    return records[-limit:][::-1]


@app.get("/api/benchmark_results")
def get_benchmark_results():
    path = "benchmark_results.json"
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="No benchmark results found. Run benchmark.py first.")
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read benchmark results: {e}")


@app.get("/api/evaluation_results")
def get_evaluation_results():
    path = "semantic_evaluation_results.json"
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="No evaluation results found. Run evaluate_semantic_pooling.py first.")
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read evaluation results: {e}")


@app.get("/api/prompt_results")
def get_prompt_results():
    path = "semantic_prompt_results.json"
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="No prompt run results found. Run run_semantic_prompts.py first.")
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read prompt results: {e}")


@app.get("/api/request/{request_id}")
def get_request_by_id(request_id: str):
    # 1. DynamoDB
    if dynamodb_telemetry_store.is_available:
        item = dynamodb_telemetry_store.get_by_id(request_id)
        if item:
            return item

    # 2. Local File
    if not os.path.exists(LOG_FILE_PATH):
        raise HTTPException(status_code=404, detail="No telemetry logs found.")
    with open(LOG_FILE_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line_str = line.strip()
            if line_str:
                try:
                    record = json.loads(line_str)
                    if record.get("request_id") == request_id:
                        return record
                except Exception:
                    pass
    raise HTTPException(status_code=404, detail=f"Request '{request_id}' not found.")


@app.post("/clear_cache")
def clear_cache(prompt: Optional[str] = Query(default=None, description="Specific prompt to evict from cache")):
    cleared_count = two_tier_cache.clear(prompt)
    return {
        "status": "cleared",
        "cleared_prompt": prompt,
        "cleared_count": cleared_count,
        "remaining_entries": len(two_tier_cache.entries)
    }


@app.post("/generate", response_model=GenerateResponse)
async def generate(
    request: GenerateRequest,
    enable_cache: bool = Query(default=True),
    enable_semantic_cache: bool = Query(default=True),
    similarity_threshold: float = Query(default=0.90, ge=0.50, le=1.00),
    cache_ttl: float = Query(default=DEFAULT_CACHE_TTL_SECONDS, ge=1.0)
):
    clean_prompt = request.prompt.strip()
    if not clean_prompt:
        raise HTTPException(status_code=400, detail="Prompt cannot be empty.")

    request_id = f"req-{uuid.uuid4().hex[:8]}"
    arrival_time = time.time()

    # Tier 1: Exact Match Cache (L1 in-memory + L2 DynamoDB)
    if enable_cache:
        exact_entry = two_tier_cache.lookup_exact(clean_prompt, request.max_new_tokens, max_age_seconds=cache_ttl)
        if exact_entry is not None:
            return _record_and_build_response(
                request_id=request_id,
                prompt=request.prompt,
                generated_text=exact_entry.response_text,
                arrival_time=arrival_time,
                proc_time_ms=0.0,
                cache_hit=True,
                match_type="exact",
                similarity_score=1.0,
                matched_prompt=exact_entry.prompt,
                prompt_tokens=exact_entry.prompt_tokens,
                generated_tokens=exact_entry.generated_tokens,
                total_model_tokens=0,
                enable_cache=enable_cache,
                enable_semantic_cache=enable_semantic_cache,
                similarity_threshold=similarity_threshold
            )

    # Tier 2: Semantic Similarity Cache (L1 in-memory + L2 DynamoDB)
    prompt_emb: Optional[list[float]] = None
    loop = asyncio.get_running_loop()

    has_entries = len(two_tier_cache.entries) > 0 or (
        two_tier_cache.dynamodb_store and two_tier_cache.dynamodb_store.is_available
    )

    if enable_cache and enable_semantic_cache and has_entries:
        try:
            prompt_emb = await loop.run_in_executor(None, compute_embedding, clean_prompt)
            if prompt_emb is not None:
                sem_entry, sim_score = two_tier_cache.lookup_semantic(
                    prompt_emb,
                    request.max_new_tokens,
                    similarity_threshold=similarity_threshold,
                    max_age_seconds=cache_ttl
                )
                if sem_entry is not None:
                    return _record_and_build_response(
                        request_id=request_id,
                        prompt=request.prompt,
                        generated_text=sem_entry.response_text,
                        arrival_time=arrival_time,
                        proc_time_ms=0.0,
                        cache_hit=True,
                        match_type="semantic",
                        similarity_score=sim_score,
                        matched_prompt=sem_entry.prompt,
                        prompt_tokens=sem_entry.prompt_tokens,
                        generated_tokens=sem_entry.generated_tokens,
                        total_model_tokens=0,
                        enable_cache=enable_cache,
                        enable_semantic_cache=enable_semantic_cache,
                        similarity_threshold=similarity_threshold
                    )
        except Exception as e:
            logger.warning("Semantic match lookup failed for '%s': %s", clean_prompt, e)

    # Full Cache Miss - Model Inference
    try:
        proc_res = await async_process_prompt(clean_prompt, request.max_new_tokens)
    except Exception as err:
        raise HTTPException(status_code=500, detail=f"Model execution error: {err}")

    gen_text = proc_res["generated_text"]
    proc_time_ms = proc_res["processing_time_sec"] * 1000.0

    if enable_cache:
        two_tier_cache.insert(
            prompt=clean_prompt,
            max_new_tokens=request.max_new_tokens,
            response_text=gen_text,
            embedding=prompt_emb,
            prompt_tokens=proc_res["prompt_tokens"],
            generated_tokens=proc_res["generated_tokens"],
            total_model_tokens=proc_res["total_model_tokens"]
        )

    return _record_and_build_response(
        request_id=request_id,
        prompt=request.prompt,
        generated_text=gen_text,
        arrival_time=arrival_time,
        proc_time_ms=proc_time_ms,
        cache_hit=False,
        match_type="none",
        similarity_score=None,
        matched_prompt=None,
        prompt_tokens=proc_res["prompt_tokens"],
        generated_tokens=proc_res["generated_tokens"],
        total_model_tokens=proc_res["total_model_tokens"],
        enable_cache=enable_cache,
        enable_semantic_cache=enable_semantic_cache,
        similarity_threshold=similarity_threshold
    )


# AWS Lambda entrypoint adapter
handler = Mangum(app, lifespan="off")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8004)
