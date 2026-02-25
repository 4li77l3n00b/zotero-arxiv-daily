import math
import re
from openai import OpenAI
from openai import BadRequestError, APIStatusError
from paper import ArxivPaper
from datetime import datetime

def _embed_texts(client: OpenAI, model: str, texts: list[str], batch_size: int = 32) -> list[list[float]]:
    def parse_max_batch_size(message: str) -> int | None:
        match = re.search(r"maximum allowed batch size\s*(\d+)", message)
        if not match:
            return None
        return int(match.group(1))

    embeddings = []
    i = 0
    dynamic_batch_size = max(1, batch_size)
    while i < len(texts):
        batch = texts[i:i + dynamic_batch_size]
        try:
            response = client.embeddings.create(model=model, input=batch)
            embeddings.extend([item.embedding for item in response.data])
            i += len(batch)
        except APIStatusError as e:
            err_msg = str(e)
            if e.status_code == 413:
                max_batch = parse_max_batch_size(err_msg)
                if max_batch is not None and max_batch < dynamic_batch_size:
                    dynamic_batch_size = max(1, max_batch)
                    continue
                if dynamic_batch_size > 1:
                    dynamic_batch_size = max(1, dynamic_batch_size // 2)
                    continue
            raise
        except BadRequestError as e:
            err_msg = str(e)
            if "Model does not exist" in err_msg or "code': 20012" in err_msg or 'code": 20012' in err_msg:
                raise ValueError(
                    f"Embedding model '{model}' is not available for current provider/base_url. "
                    "Please set EMBEDDING_MODEL to a supported one, e.g. BAAI/bge-large-zh-v1.5 for SiliconFlow."
                ) from e
            raise ValueError(
                f"Embedding request failed for model '{model}': {e}"
            ) from e
    return embeddings


def _cosine_similarity(vec_a: list[float], vec_b: list[float], norm_a: float, norm_b: float) -> float:
    if norm_a == 0 or norm_b == 0:
        return 0.0
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    return dot / (norm_a * norm_b)


def rerank_paper(
    candidate: list[ArxivPaper],
    corpus: list[dict],
    api_key: str,
    base_url: str = None,
    model: str = "netease-youdao/bce-embedding-base_v1",
) -> list[ArxivPaper]:
    if len(candidate) == 0:
        return candidate

    if len(corpus) == 0:
        for c in candidate:
            c.score = 0.0
        return candidate

    client = OpenAI(api_key=api_key, base_url=base_url)

    # sort corpus by date, from newest to oldest
    corpus = sorted(corpus, key=lambda x: datetime.strptime(x['data']['dateAdded'], '%Y-%m-%dT%H:%M:%SZ'), reverse=True)

    time_decay_weight = [1 / (1 + math.log10(i + 1)) for i in range(len(corpus))]
    weight_sum = sum(time_decay_weight)
    time_decay_weight = [w / weight_sum for w in time_decay_weight]

    corpus_texts = [paper['data']['abstractNote'] for paper in corpus]
    candidate_texts = [paper.summary for paper in candidate]
    corpus_feature = _embed_texts(client, model, corpus_texts)
    candidate_feature = _embed_texts(client, model, candidate_texts)

    corpus_norms = [math.sqrt(sum(x * x for x in vec)) for vec in corpus_feature]

    scores = []
    for cand_vec in candidate_feature:
        cand_norm = math.sqrt(sum(x * x for x in cand_vec))
        similarities = [
            _cosine_similarity(cand_vec, corp_vec, cand_norm, corp_norm)
            for corp_vec, corp_norm in zip(corpus_feature, corpus_norms)
        ]
        score = sum(sim * w for sim, w in zip(similarities, time_decay_weight)) * 10
        scores.append(score)

    for s, c in zip(scores, candidate):
        c.score = float(s)

    candidate = sorted(candidate, key=lambda x: x.score, reverse=True)
    return candidate