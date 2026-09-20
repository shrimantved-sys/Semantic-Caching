"""AWS DynamoDB Storage and Embedding Provider for Serverless Two-Tier LLM Cache.

Targets AWS Region: ap-southeast-2
- DynamoDB Table 'llm-semantic-cache': Persistent Tier 1 & Tier 2 Cache with TTL
- DynamoDB Table 'llm-telemetry-logs': Observability & telemetry traces
- Bedrock Runtime: 'amazon.titan-embed-text-v2:0' with fallback to HuggingFace or token vectorizer
"""

from decimal import Decimal
import hashlib
import json
import logging
import math
import os
import time
from typing import Any, Optional

import boto3
from botocore.exceptions import ClientError

logger = logging.getLogger("aws_storage")

AWS_REGION = os.environ.get("AWS_REGION", os.environ.get("AWS_DEFAULT_REGION", "ap-southeast-2"))
CACHE_TABLE_NAME = os.environ.get("CACHE_TABLE_NAME", "llm-semantic-cache")
TELEMETRY_TABLE_NAME = os.environ.get("TELEMETRY_TABLE_NAME", "llm-telemetry-logs")
BEDROCK_EMBED_MODEL = os.environ.get("BEDROCK_EMBED_MODEL", "amazon.titan-embed-text-v2:0")


def _to_decimal(val: Any) -> Any:
    """Recursively convert float types to Decimal for DynamoDB storage."""
    if isinstance(val, float):
        if math.isnan(val) or math.isinf(val):
            return None
        return Decimal(str(round(val, 6)))
    if isinstance(val, dict):
        return {k: _to_decimal(v) for k, v in val.items()}
    if isinstance(val, (list, tuple)):
        return [_to_decimal(x) for x in val]
    return val


def _from_decimal(val: Any) -> Any:
    """Recursively convert Decimal types back to standard Python float/int."""
    if isinstance(val, Decimal):
        return float(val) if val % 1 != 0 else int(val)
    if isinstance(val, dict):
        return {k: _from_decimal(v) for k, v in val.items()}
    if isinstance(val, list):
        return [_from_decimal(x) for x in val]
    return val


def make_cache_key(prompt: str, max_new_tokens: int) -> str:
    """Generate a deterministic key for Tier 1 hash lookup."""
    clean = prompt.strip().lower()
    prompt_hash = hashlib.sha256(clean.encode("utf-8")).hexdigest()[:16]
    return f"{prompt_hash}:{max_new_tokens}"


def compute_cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """Compute dot product of two L2-normalized vectors."""
    if not vec_a or not vec_b or len(vec_a) != len(vec_b):
        return 0.0
    return sum(a * b for a, b in zip(vec_a, vec_b))


class EmbeddingManager:
    """Unified embedding provider: Bedrock Titan v2 -> PyTorch HuggingFace -> N-gram vectorizer."""

    def __init__(self):
        self.provider_name = "uninitialized"
        self._bedrock_client = None
        self._hf_tokenizer = None
        self._hf_model = None
        self._init_provider()

    def _init_provider(self):
        # 1. Try AWS Bedrock in selected Region
        try:
            client = boto3.client("bedrock-runtime", region_name=AWS_REGION)
            test_body = json.dumps({"inputText": "ping", "dimensions": 256, "normalize": True})
            resp = client.invoke_model(
                modelId=BEDROCK_EMBED_MODEL,
                body=test_body,
                contentType="application/json",
                accept="application/json"
            )
            data = json.loads(resp["body"].read())
            if "embedding" in data and len(data["embedding"]) > 0:
                self._bedrock_client = client
                self.provider_name = f"bedrock:{BEDROCK_EMBED_MODEL}"
                logger.info("Using AWS Bedrock embedding provider: %s", self.provider_name)
                return
        except Exception as e:
            logger.info("AWS Bedrock not directly available or account verification pending (%s). Checking HuggingFace...", e)

        # 2. Try Local PyTorch + Transformers (sentence-transformers/all-MiniLM-L6-v2)
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
            emb_name = "sentence-transformers/all-MiniLM-L6-v2"
            self._hf_tokenizer = AutoTokenizer.from_pretrained(emb_name)
            self._hf_model = AutoModel.from_pretrained(emb_name)
            self._hf_model.eval()
            self.provider_name = f"transformers:{emb_name}"
            logger.info("Using HuggingFace Transformers embedding provider: %s", self.provider_name)
            return
        except Exception as e:
            logger.info("PyTorch/Transformers not active in this environment (%s). Using lightweight N-gram feature embedding.", e)

        # 3. Fallback: Fast, deterministic character/word N-gram semantic vectorizer
        self.provider_name = "ngram-sparse-hash"
        logger.info("Using deterministic N-gram feature embedding provider.")

    def compute_embedding(self, prompt: str) -> Optional[list[float]]:
        clean = prompt.strip()
        if not clean:
            return None

        # Provider: Bedrock
        if self._bedrock_client is not None:
            try:
                body = json.dumps({"inputText": clean, "dimensions": 256, "normalize": True})
                resp = self._bedrock_client.invoke_model(
                    modelId=BEDROCK_EMBED_MODEL,
                    body=body,
                    contentType="application/json",
                    accept="application/json"
                )
                data = json.loads(resp["body"].read())
                emb = data.get("embedding", [])
                if emb:
                    return [float(x) for x in emb]
            except Exception as e:
                logger.warning("Bedrock embedding call failed: %s", e)

        # Provider: HuggingFace
        if self._hf_tokenizer is not None and self._hf_model is not None:
            try:
                import torch
                encoded = self._hf_tokenizer([clean], padding=True, truncation=True, return_tensors="pt")
                with torch.no_grad():
                    outputs = self._hf_model(**encoded)
                    token_embeddings = outputs[0]
                    mask = encoded["attention_mask"].unsqueeze(-1).expand(token_embeddings.size()).float()
                    sum_embeddings = torch.sum(token_embeddings * mask, dim=1)
                    sum_mask = torch.clamp(mask.sum(dim=1), min=1e-9)
                    mean_pooled = sum_embeddings / sum_mask
                    norm_emb = torch.nn.functional.normalize(mean_pooled, p=2, dim=1)[0]
                    return [float(x) for x in norm_emb.tolist()]
            except Exception as e:
                logger.warning("HuggingFace embedding computation failed: %s", e)

        # Fallback: Deterministic 256-dimensional N-gram hash vector with L2 normalization
        return self._compute_ngram_vector(clean, dims=256)

    @staticmethod
    def _compute_ngram_vector(text: str, dims: int = 256) -> list[float]:
        words = text.lower().split()
        vec = [0.0] * dims
        for word in words:
            # Word token hash
            idx = int(hashlib.md5(word.encode("utf-8")).hexdigest(), 16) % dims
            vec[idx] += 1.5
            # Character trigram hashes
            padded = f"^{word}$"
            for i in range(max(1, len(padded) - 2)):
                tri = padded[i:i + 3]
                t_idx = int(hashlib.sha1(tri.encode("utf-8")).hexdigest(), 16) % dims
                vec[t_idx] += 1.0

        norm = math.sqrt(sum(x * x for x in vec))
        if norm > 1e-9:
            return [x / norm for x in vec]
        return vec


class DynamoDBCacheStore:
    """Manages persistent cache entries in Amazon DynamoDB with automatic TTL."""

    def __init__(self, table_name: str = CACHE_TABLE_NAME, region_name: str = AWS_REGION):
        self.table_name = table_name
        self.region_name = region_name
        self._dynamodb = boto3.resource("dynamodb", region_name=region_name)
        self.table = self._dynamodb.Table(table_name)
        self.is_available = True
        self._test_connection()

    def _test_connection(self):
        try:
            self.table.load()
        except ClientError as e:
            logger.warning("DynamoDB cache table '%s' not accessible: %s. Operating in memory fallback.", self.table_name, e)
            self.is_available = False

    def get_exact(self, prompt: str, max_new_tokens: int, max_age_seconds: float) -> Optional[dict]:
        if not self.is_available:
            return None
        cache_key = make_cache_key(prompt, max_new_tokens)
        try:
            resp = self.table.get_item(Key={"cache_key": cache_key})
            item = resp.get("Item")
            if not item:
                return None
            item = _from_decimal(item)
            now = time.time()
            if (now - item.get("timestamp", 0)) > max_age_seconds:
                return None
            return item
        except Exception as e:
            logger.error("DynamoDB get_exact failed for key %s: %s", cache_key, e)
            return None

    def get_candidates(self, max_new_tokens: int, max_age_seconds: float) -> list[dict]:
        """Scan candidate entries for Tier 2 semantic matching."""
        if not self.is_available:
            return []
        try:
            now = time.time()
            min_timestamp = now - max_age_seconds
            resp = self.table.scan(
                FilterExpression="#tok = :tok AND #ts >= :min_ts",
                ExpressionAttributeNames={"#tok": "max_new_tokens", "#ts": "timestamp"},
                ExpressionAttributeValues={":tok": max_new_tokens, ":min_ts": Decimal(str(int(min_timestamp)))},
                Limit=100
            )
            items = resp.get("Items", [])
            return [_from_decimal(item) for item in items if item.get("embedding")]
        except Exception as e:
            logger.error("DynamoDB get_candidates scan failed: %s", e)
            return []

    def put_entry(
        self,
        prompt: str,
        max_new_tokens: int,
        response_text: str,
        embedding: Optional[list[float]],
        prompt_tokens: int = 0,
        generated_tokens: int = 0,
        total_model_tokens: int = 0,
        ttl_seconds: float = 3600.0
    ) -> bool:
        if not self.is_available:
            return False
        clean = prompt.strip()
        cache_key = make_cache_key(clean, max_new_tokens)
        now = time.time()
        ttl_timestamp = int(now + ttl_seconds)

        item = {
            "cache_key": cache_key,
            "prompt": clean,
            "max_new_tokens": max_new_tokens,
            "response_text": response_text,
            "embedding": embedding,
            "timestamp": now,
            "ttl_timestamp": ttl_timestamp,
            "prompt_tokens": prompt_tokens,
            "generated_tokens": generated_tokens,
            "total_model_tokens": total_model_tokens
        }

        try:
            self.table.put_item(Item=_to_decimal(item))
            return True
        except Exception as e:
            logger.error("DynamoDB put_entry failed for key %s: %s", cache_key, e)
            return False

    def clear(self, prompt: Optional[str] = None) -> int:
        if not self.is_available:
            return 0
        try:
            if prompt:
                clean = prompt.strip()
                resp = self.table.scan(
                    FilterExpression="#p = :p",
                    ExpressionAttributeNames={"#p": "prompt"},
                    ExpressionAttributeValues={":p": clean}
                )
                items = resp.get("Items", [])
                with self.table.batch_writer() as batch:
                    for item in items:
                        batch.delete_item(Key={"cache_key": item["cache_key"]})
                return len(items)

            # Clear all entries
            resp = self.table.scan(ProjectionExpression="cache_key")
            items = resp.get("Items", [])
            with self.table.batch_writer() as batch:
                for item in items:
                    batch.delete_item(Key={"cache_key": item["cache_key"]})
            return len(items)
        except Exception as e:
            logger.error("DynamoDB clear cache failed: %s", e)
            return 0

    def list_entries(self, limit: int = 50) -> list[dict]:
        if not self.is_available:
            return []
        try:
            resp = self.table.scan(Limit=limit)
            items = resp.get("Items", [])
            now = time.time()
            results = []
            for item in items:
                decoded = _from_decimal(item)
                ts = decoded.get("timestamp", now)
                results.append({
                    "prompt": decoded.get("prompt", ""),
                    "max_new_tokens": decoded.get("max_new_tokens", 0),
                    "response_preview": decoded.get("response_text", "")[:80] + ("..." if len(decoded.get("response_text", "")) > 80 else ""),
                    "timestamp": ts,
                    "age_seconds": round(now - ts, 1),
                    "has_embedding": decoded.get("embedding") is not None
                })
            return sorted(results, key=lambda x: x["timestamp"], reverse=True)
        except Exception as e:
            logger.error("DynamoDB list_entries failed: %s", e)
            return []


class DynamoDBTelemetryStore:
    """Persists and queries inference telemetry traces in Amazon DynamoDB."""

    def __init__(self, table_name: str = TELEMETRY_TABLE_NAME, region_name: str = AWS_REGION):
        self.table_name = table_name
        self.region_name = region_name
        self._dynamodb = boto3.resource("dynamodb", region_name=region_name)
        self.table = self._dynamodb.Table(table_name)
        self.is_available = True
        self._test_connection()

    def _test_connection(self):
        try:
            self.table.load()
        except ClientError as e:
            logger.warning("DynamoDB telemetry table '%s' not accessible: %s", self.table_name, e)
            self.is_available = False

    def append_log(self, record: dict) -> bool:
        if not self.is_available:
            return False
        try:
            self.table.put_item(Item=_to_decimal(record))
            return True
        except Exception as e:
            logger.error("DynamoDB append_log failed for request %s: %s", record.get("request_id"), e)
            return False

    def get_logs(self, limit: int = 50) -> list[dict]:
        if not self.is_available:
            return []
        try:
            resp = self.table.scan(Limit=limit)
            items = [_from_decimal(x) for x in resp.get("Items", [])]
            return sorted(items, key=lambda x: x.get("timestamp", ""), reverse=True)[:limit]
        except Exception as e:
            logger.error("DynamoDB get_logs failed: %s", e)
            return []

    def get_by_id(self, request_id: str) -> Optional[dict]:
        if not self.is_available:
            return None
        try:
            resp = self.table.get_item(Key={"request_id": request_id})
            item = resp.get("Item")
            return _from_decimal(item) if item else None
        except Exception as e:
            logger.error("DynamoDB get_by_id failed for %s: %s", request_id, e)
            return None
