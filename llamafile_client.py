import requests
import settings
import numpy as np


def tokenize(
    text: str,
    base_url_prefix: str = "http://localhost",
) -> list[int]:
    response = requests.post(
        url=f"{base_url_prefix}:{settings.EMBEDDING_MODEL_PORT}/tokenize",
        headers={"Content-Type": "application/json"},
        json={
            "content": text,
        },
    )
    response.raise_for_status()
    return response.json()["tokens"]


def detokenize(
    tokens: list[int],
    base_url_prefix: str = "http://localhost",
) -> str:
    response = requests.post(
        url=f"{base_url_prefix}:{settings.EMBEDDING_MODEL_PORT}/detokenize",
        headers={"Content-Type": "application/json"},
        json={
            "tokens": tokens,
        },
    )
    response.raise_for_status()
    return response.json()["content"]


def embed(text: str, base_url_prefix: str = "http://localhost") -> np.ndarray:
    response = requests.post(
        url=f"{base_url_prefix}:{settings.EMBEDDING_MODEL_PORT}/embedding",
        headers={"Content-Type": "application/json"},
        json={
            "content": text,
        },
    )
    response.raise_for_status()
    emb = np.array(response.json()["embedding"], dtype=np.float32)
    return np.expand_dims(emb, axis=0)


def completion(prompt: str, base_url_prefix: str = "http://localhost", **kwargs):
    # defaults
    options = {
        "temperature": 0,
        "seed": 0,
    }
    options.update(kwargs)

    response = requests.post(
        url=f"{base_url_prefix}:{settings.GENERATION_MODEL_PORT}/completion",
        headers={"Content-Type": "application/json"},
        json={"prompt": prompt, **options},
    )
    response.raise_for_status()
    return response.json()["content"]
