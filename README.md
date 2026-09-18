# Semantic-Caching
#  Two-Tier Semantic LLM Cache & Inference Engine
> **Cut LLM latency from 500ms down to 0.01ms and slash token bills by 60–80% without sacrificing response quality.**

## The Problem Every LLM Developer Hits

If you have ever shipped an LLM app to production, you know the sinking feeling of watching your API bill climb while your users complain about 500ms to 2-second response delays.

When we looked at real-world production query traffic, an obvious pattern emerged:
- **A huge portion of user queries are either identical or semantic variations of questions already asked.**
- Questions like:
  - *"What is the boiling point of water at sea level?"*
  - *"At what temperature does water boil at sea level?"*
  - *"How hot does water need to be to boil?"*
  - 
In a standard setup, every single one of those questions triggers a brand-new cloud API call to an LLM. You pay for the prompt tokens, you pay for the generation tokens, and your user sits waiting while the model generates the exact same answer it generated thirty seconds ago for someone else.

Traditional web caches (like Redis exact-key lookups) fail completely here because string hashing is brittle: one typo, a slight paraphrase, or a flipped word order results in a 100% cache miss.

**This project solves that problem.**

We built an inference server with a **Two-Tier Cache Architecture**:
1. **Tier 1 (Exact String Match)**: Sub-millisecond $O(1)$ memory lookup for identical queries.
2. **Tier 2 (Semantic Cosine Match)**: Local sentence embedding comparison that recognizes paraphrases and synonyms, serving cached answers in under 15ms without touching the cloud LLM.
3. **Cache Miss (Direct Inference)**: Direct, queue-free execution via **Groq's LPU hardware** (`qwen/qwen3.8-27b`), automatically populating the cache for future queries.

