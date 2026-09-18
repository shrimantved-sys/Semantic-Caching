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
from pydantic import BaseModel, Field
import torch
from transformers import AutoModel, AutoTokenizer

# Load environment variables from .env file
load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("inference_server")

# Configuration
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "").strip()
GROQ_MODEL_NAME = os.environ.get("GROQ_MODEL_NAME", "qwen/qwen3.8-27b")
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
MAX_BATCH_SIZE = 4
BATCH_TIMEOUT = 0.2
LOG_FILE_PATH = "telemetry_metrics.jsonl"
DEFAULT_CACHE_TTL_SECONDS = 3600.0
AUTO_SHUTDOWN_SECONDS = float(os.environ.get("AUTO_SHUTDOWN_SECONDS", "0"))
INDEX_HTML_PATH = "index.html" if os.path.exists("index.html") else os.path.join("static", "index.html")

# Embedding model for Tier 2 semantic matching
logger.info("Loading embedding model '%s'...", EMBEDDING_MODEL_NAME)
emb_tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL_NAME)
emb_model = AutoModel.from_pretrained(EMBEDDING_MODEL_NAME)
emb_model.eval()


def compute_embedding(prompt: str) -> Optional[torch.Tensor]:
    """Compute normalized mean-pooled sentence embedding for a prompt."""
    clean = prompt.strip()
    if not clean:
        return None
    try:
        encoded = emb_tokenizer([clean], padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            outputs = emb_model(**encoded)
            token_embeddings = outputs[0]
            mask = encoded["attention_mask"].unsqueeze(-1).expand(token_embeddings.size()).float()
            sum_embeddings = torch.sum(token_embeddings * mask, dim=1)
            sum_mask = torch.clamp(mask.sum(dim=1), min=1e-9)
            mean_pooled = sum_embeddings / sum_mask
            return torch.nn.functional.normalize(mean_pooled, p=2, dim=1)[0]
    except Exception as e:
        logger.error("Failed to compute embedding for prompt: %r. Error: %s", prompt, e)
        return None


def compute_cosine_similarity(vec_a: torch.Tensor, vec_b: torch.Tensor) -> float:
    return float(torch.dot(vec_a, vec_b).item())


@dataclass(slots=True)
class CacheEntry:
    prompt: str
    max_new_tokens: int
    response_text: str
    embedding: Optional[torch.Tensor]
    timestamp: float
    prompt_tokens: int = 0
    generated_tokens: int = 0
    total_model_tokens: int = 0

    def is_expired(self, max_age_seconds: float) -> bool:
        return (time.time() - self.timestamp) > max_age_seconds


class TwoTierCache:
    """Two-tier cache combining exact string lookup (Tier 1) and semantic similarity (Tier 2)."""

    def __init__(self, default_ttl_seconds: float = DEFAULT_CACHE_TTL_SECONDS):
        self.default_ttl_seconds = default_ttl_seconds
        self.entries: dict[tuple[str, int], CacheEntry] = {}

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
        return self.entries.get((prompt.strip(), max_new_tokens))

    def lookup_semantic(
        self,
        prompt_embedding: torch.Tensor,
        max_new_tokens: int,
        similarity_threshold: float,
        max_age_seconds: Optional[float] = None
    ) -> tuple[Optional[CacheEntry], Optional[float]]:
        self._evict_expired(max_age_seconds)

        candidates = [
            entry for (_, tok), entry in self.entries.items()
            if tok == max_new_tokens and entry.embedding is not None
        ]
        if not candidates:
            return None, None

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

        return None, round(best_sim, 4) if best_entry is not None else None

    def insert(
        self,
        prompt: str,
        max_new_tokens: int,
        response_text: str,
        embedding: Optional[torch.Tensor] = None,
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
        self.entries[(clean, max_new_tokens)] = entry
        return entry

    def clear(self, prompt: Optional[str] = None) -> int:
        if prompt:
            clean = prompt.strip()
            keys = [k for k in self.entries if k[0] == clean]
            for k in keys:
                del self.entries[k]
            return len(keys)
        count = len(self.entries)
        self.entries.clear()
        return count


@dataclass(slots=True)
class RequestItem:
    request_id: str
    prompt: str
    max_new_tokens: int
    future: asyncio.Future
    arrival_time: float
    embedding: Optional[torch.Tensor] = None


@dataclass(slots=True)
class BatchResult:
    generated_texts: list[str]
    processing_time_sec: float
    prompt_tokens_list: list[int]
    padded_tokens_list: list[int]
    generated_tokens_list: list[int]
    total_model_tokens_list: list[int]


# App State
two_tier_cache = TwoTierCache()
request_queue: asyncio.Queue[RequestItem] = asyncio.Queue()
http_client: Optional[httpx.AsyncClient] = None


def append_telemetry_log(record: dict) -> None:
    with open(LOG_FILE_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


async def async_process_batch(prompts: list[str], max_new_tokens: int) -> BatchResult:
    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY is not configured. Please set GROQ_API_KEY in your .env file or environment variables.")
    start_time = time.time()
    batch_size = len(prompts)
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    client = http_client or httpx.AsyncClient(timeout=30.0)

    async def fetch_one(prompt: str) -> dict:
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
        return resp.json()

    api_responses = await asyncio.gather(*[fetch_one(p) for p in prompts])

    generated_texts = []
    prompt_tokens_list = []
    generated_tokens_list = []
    total_model_tokens_list = []

    for r in api_responses:
        text = r["choices"][0]["message"]["content"] or ""
        usage = r.get("usage", {})
        p_tok = usage.get("prompt_tokens", len(text.split()))
        g_tok = usage.get("completion_tokens", len(text.split()))
        t_tok = usage.get("total_tokens", p_tok + g_tok)

        generated_texts.append(text)
        prompt_tokens_list.append(p_tok)
        generated_tokens_list.append(g_tok)
        total_model_tokens_list.append(t_tok)

    return BatchResult(
        generated_texts=generated_texts,
        processing_time_sec=time.time() - start_time,
        prompt_tokens_list=prompt_tokens_list,
        padded_tokens_list=[0] * batch_size,
        generated_tokens_list=generated_tokens_list,
        total_model_tokens_list=total_model_tokens_list
    )


async def batch_worker():
    while True:
        first_item = await request_queue.get()
        batch: list[RequestItem] = [first_item]
        deadline = time.time() + BATCH_TIMEOUT

        while len(batch) < MAX_BATCH_SIZE:
            remaining = deadline - time.time()
            if remaining <= 0:
                break
            try:
                item = await asyncio.wait_for(request_queue.get(), timeout=remaining)
                batch.append(item)
            except asyncio.TimeoutError:
                break

        batch_size = len(batch)
        prompts = [item.prompt for item in batch]
        max_tokens = max(item.max_new_tokens for item in batch)

        try:
            batch_res = await async_process_batch(prompts, max_tokens)
            for i, item in enumerate(batch):
                gen_text = batch_res.generated_texts[i]
                two_tier_cache.insert(
                    prompt=item.prompt,
                    max_new_tokens=item.max_new_tokens,
                    response_text=gen_text,
                    embedding=item.embedding,
                    prompt_tokens=batch_res.prompt_tokens_list[i],
                    generated_tokens=batch_res.generated_tokens_list[i],
                    total_model_tokens=batch_res.total_model_tokens_list[i]
                )
                if not item.future.done():
                    item.future.set_result({
                        "generated_text": gen_text,
                        "processing_time_sec": batch_res.processing_time_sec,
                        "batch_size": batch_size,
                        "prompt_tokens": batch_res.prompt_tokens_list[i],
                        "padded_tokens": batch_res.padded_tokens_list[i],
                        "generated_tokens": batch_res.generated_tokens_list[i],
                        "total_model_tokens": batch_res.total_model_tokens_list[i]
                    })
        except Exception as e:
            logger.error("Error processing batch: %s", e, exc_info=True)
            for item in batch:
                if not item.future.done():
                    item.future.set_exception(e)
        finally:
            for _ in range(batch_size):
                request_queue.task_done()


@asynccontextmanager
async def lifespan(app: FastAPI):
    global http_client
    http_client = httpx.AsyncClient(timeout=30.0)
    worker_task = asyncio.create_task(batch_worker())
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
        worker_task.cancel()
        await asyncio.gather(worker_task, *([shutdown_task] if shutdown_task else []), return_exceptions=True)
        if http_client is not None:
            await http_client.aclose()


app = FastAPI(
    title="LLM Inference Server - Two-Tier Cache Architecture",
    description="Inference server with Tier 1 exact match cache, Tier 2 semantic embedding cache, and dynamic batching.",
    lifespan=lifespan
)


# Request / Response Schemas
class GenerateRequest(BaseModel):
    prompt: str = Field(..., description="Input prompt text", min_length=1)
    max_new_tokens: int = Field(default=20, ge=1, le=100)


class BatchGenerateRequest(BaseModel):
    prompts: list[str] = Field(..., description="List of prompt strings", min_length=1)
    max_new_tokens: int = Field(default=20, ge=1, le=100)


class GenerateResponse(BaseModel):
    request_id: str
    prompt: str
    generated_text: str
    queue_wait_time_ms: float
    processing_time_ms: float
    total_latency_ms: float
    batch_size: int
    cache_hit: bool
    match_type: str = Field(default="none", description="Match type: 'exact', 'semantic', or 'none'")
    similarity_score: Optional[float] = Field(default=None, description="Similarity score for semantic match")
    matched_prompt: Optional[str] = Field(default=None, description="Original prompt matched against")
    prompt_tokens: int
    generated_tokens: int
    padded_tokens: int
    total_model_tokens: int


def _record_and_build_response(
    request_id: str,
    prompt: str,
    generated_text: str,
    arrival_time: float,
    queue_wait_ms: float,
    proc_time_ms: float,
    batch_size: int,
    cache_hit: bool,
    match_type: str,
    similarity_score: Optional[float],
    matched_prompt: Optional[str],
    prompt_tokens: int,
    generated_tokens: int,
    padded_tokens: int,
    total_model_tokens: int,
    enable_batching: bool,
    enable_cache: bool,
    enable_semantic_cache: bool,
    similarity_threshold: float
) -> GenerateResponse:
    total_latency_ms = (time.time() - arrival_time) * 1000.0

    response = GenerateResponse(
        request_id=request_id,
        prompt=prompt,
        generated_text=generated_text,
        queue_wait_time_ms=round(max(queue_wait_ms, 0.0), 2),
        processing_time_ms=round(proc_time_ms, 2),
        total_latency_ms=round(total_latency_ms, 2),
        batch_size=batch_size,
        cache_hit=cache_hit,
        match_type=match_type,
        similarity_score=similarity_score,
        matched_prompt=matched_prompt,
        prompt_tokens=prompt_tokens,
        generated_tokens=generated_tokens,
        padded_tokens=padded_tokens,
        total_model_tokens=total_model_tokens
    )

    append_telemetry_log({
        "request_id": request_id,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(arrival_time)),
        "prompt": prompt,
        "generated_text": generated_text,
        "prompt_char_len": len(prompt.strip()),
        "max_new_tokens": total_model_tokens,
        "queue_time_ms": round(max(queue_wait_ms, 0.0), 2),
        "processing_time_ms": round(proc_time_ms, 2),
        "total_latency_ms": round(total_latency_ms, 2),
        "batch_size": batch_size,
        "cache_hit": cache_hit,
        "match_type": match_type,
        "similarity_score": similarity_score,
        "matched_prompt": matched_prompt,
        "prompt_tokens": prompt_tokens,
        "generated_tokens": generated_tokens,
        "padded_tokens": padded_tokens,
        "total_model_tokens": total_model_tokens,
        "batching_enabled": enable_batching,
        "cache_enabled": enable_cache,
        "semantic_cache_enabled": enable_semantic_cache,
        "similarity_threshold": similarity_threshold
    })

    return response


# Endpoints
@app.get("/")
@app.get("/playground")
def playground():
    return FileResponse(INDEX_HTML_PATH)


@app.get("/api/status")
def status_api():
    return {
        "status": "ok",
        "stage": "Stage E: Two-Tier Cache Architecture (Tier 1 Exact + Tier 2 Semantic)",
        "generative_model": GROQ_MODEL_NAME,
        "log_file": LOG_FILE_PATH,
        "cache_entries": len(two_tier_cache.entries),
        "cache_ttl_seconds": two_tier_cache.default_ttl_seconds
    }


@app.get("/api/cache_entries")
def get_cache_entries():
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


@app.post("/generate_batch", response_model=list[GenerateResponse])
async def generate_batch(
    request: BatchGenerateRequest,
    enable_batching: bool = Query(default=True),
    enable_cache: bool = Query(default=True),
    enable_semantic_cache: bool = Query(default=True),
    similarity_threshold: float = Query(default=0.90, ge=0.50, le=1.00),
    cache_ttl: float = Query(default=DEFAULT_CACHE_TTL_SECONDS, ge=1.0)
):
    tasks = [
        generate(
            GenerateRequest(prompt=p, max_new_tokens=request.max_new_tokens),
            enable_batching=enable_batching,
            enable_cache=enable_cache,
            enable_semantic_cache=enable_semantic_cache,
            similarity_threshold=similarity_threshold,
            cache_ttl=cache_ttl
        )
        for p in request.prompts
    ]
    return await asyncio.gather(*tasks)


@app.post("/generate", response_model=GenerateResponse)
async def generate(
    request: GenerateRequest,
    enable_batching: bool = Query(default=True),
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

    # Tier 1: Exact Match Cache
    if enable_cache:
        exact_entry = two_tier_cache.lookup_exact(clean_prompt, request.max_new_tokens, max_age_seconds=cache_ttl)
        if exact_entry is not None:
            return _record_and_build_response(
                request_id=request_id,
                prompt=request.prompt,
                generated_text=exact_entry.response_text,
                arrival_time=arrival_time,
                queue_wait_ms=0.0,
                proc_time_ms=0.0,
                batch_size=0,
                cache_hit=True,
                match_type="exact",
                similarity_score=1.0,
                matched_prompt=exact_entry.prompt,
                prompt_tokens=exact_entry.prompt_tokens,
                generated_tokens=exact_entry.generated_tokens,
                padded_tokens=0,
                total_model_tokens=0,
                enable_batching=enable_batching,
                enable_cache=enable_cache,
                enable_semantic_cache=enable_semantic_cache,
                similarity_threshold=similarity_threshold
            )

    # Tier 2: Semantic Similarity Cache
    prompt_emb: Optional[torch.Tensor] = None
    loop = asyncio.get_running_loop()

    if enable_cache and enable_semantic_cache and len(two_tier_cache.entries) > 0:
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
                        queue_wait_ms=0.0,
                        proc_time_ms=0.0,
                        batch_size=0,
                        cache_hit=True,
                        match_type="semantic",
                        similarity_score=sim_score,
                        matched_prompt=sem_entry.prompt,
                        prompt_tokens=sem_entry.prompt_tokens,
                        generated_tokens=sem_entry.generated_tokens,
                        padded_tokens=0,
                        total_model_tokens=0,
                        enable_batching=enable_batching,
                        enable_cache=enable_cache,
                        enable_semantic_cache=enable_semantic_cache,
                        similarity_threshold=similarity_threshold
                    )
        except Exception as e:
            logger.warning("Semantic match lookup failed for '%s': %s", clean_prompt, e)

    # Full Cache Miss - Model Inference
    if not enable_batching:
        start_proc = time.time()
        single_res = await async_process_batch([clean_prompt], request.max_new_tokens)
        proc_time_ms = (time.time() - start_proc) * 1000.0
        gen_text = single_res.generated_texts[0]

        if enable_cache:
            two_tier_cache.insert(
                prompt=clean_prompt,
                max_new_tokens=request.max_new_tokens,
                response_text=gen_text,
                embedding=prompt_emb,
                prompt_tokens=single_res.prompt_tokens_list[0],
                generated_tokens=single_res.generated_tokens_list[0],
                total_model_tokens=single_res.total_model_tokens_list[0]
            )

        return _record_and_build_response(
            request_id=request_id,
            prompt=request.prompt,
            generated_text=gen_text,
            arrival_time=arrival_time,
            queue_wait_ms=0.0,
            proc_time_ms=proc_time_ms,
            batch_size=1,
            cache_hit=False,
            match_type="none",
            similarity_score=None,
            matched_prompt=None,
            prompt_tokens=single_res.prompt_tokens_list[0],
            generated_tokens=single_res.generated_tokens_list[0],
            padded_tokens=0,
            total_model_tokens=single_res.total_model_tokens_list[0],
            enable_batching=False,
            enable_cache=enable_cache,
            enable_semantic_cache=enable_semantic_cache,
            similarity_threshold=similarity_threshold
        )

    # Dynamic Batching
    future = loop.create_future()
    item = RequestItem(
        request_id=request_id,
        prompt=clean_prompt,
        max_new_tokens=request.max_new_tokens,
        future=future,
        arrival_time=arrival_time,
        embedding=prompt_emb
    )
    await request_queue.put(item)

    try:
        res_data = await future
    except Exception as err:
        raise HTTPException(status_code=500, detail=f"Model execution error: {err}")

    total_latency_ms = (time.time() - arrival_time) * 1000.0
    proc_time_ms = res_data["processing_time_sec"] * 1000.0
    queue_wait_ms = total_latency_ms - proc_time_ms

    return _record_and_build_response(
        request_id=request_id,
        prompt=request.prompt,
        generated_text=res_data["generated_text"],
        arrival_time=arrival_time,
        queue_wait_ms=queue_wait_ms,
        proc_time_ms=proc_time_ms,
        batch_size=res_data["batch_size"],
        cache_hit=False,
        match_type="none",
        similarity_score=None,
        matched_prompt=None,
        prompt_tokens=res_data["prompt_tokens"],
        generated_tokens=res_data["generated_tokens"],
        padded_tokens=res_data["padded_tokens"],
        total_model_tokens=res_data["total_model_tokens"],
        enable_batching=True,
        enable_cache=enable_cache,
        enable_semantic_cache=enable_semantic_cache,
        similarity_threshold=similarity_threshold
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8004)
