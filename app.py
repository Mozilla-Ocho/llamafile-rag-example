import json
import logging
import re
import traceback
from pathlib import Path
from typing import Iterator

import click
import faiss
import numpy as np
import requests
from bs4 import BeautifulSoup

import settings
import llamafile_client as llamafile

from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)


def chunk_text(text: str) -> Iterator[str]:
    if settings.INDEX_TEXT_CHUNK_LEN > 0:
        chunk_len = min(settings.INDEX_TEXT_CHUNK_LEN, settings.EMBEDDING_MODEL_MAX_LEN)
    else:
        chunk_len = settings.EMBEDDING_MODEL_MAX_LEN

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_len,
        chunk_overlap=40,
        length_function=len,
        is_separator_regex=False,
    )

    chunks = text_splitter.split_text(text)

    for chunk in chunks:
        yield chunk


def load_data_for_indexing() -> Iterator[str]:
    for url in settings.INDEX_URLS:
        try:
            response = requests.get(url)
            response.raise_for_status()
            text = BeautifulSoup(response.text, "html.parser").get_text()
            for chunk in chunk_text(text):
                yield chunk
        except Exception as e:
            traceback.print_exc()
            logger.error(f"skipping {url}: {e}")
            continue

    for directory in settings.INDEX_LOCAL_DATA_DIRS:
        for path in Path(directory).rglob("*.txt"):
            with open(path, "r") as f:
                text = f.read()
                for chunk in chunk_text(text):
                    yield chunk


def embed(text: str) -> np.ndarray:
    embedding = llamafile.embed(text)
    # why L2-normalize here?
    # see: https://github.com/facebookresearch/faiss/wiki/MetricType-and-distances#how-can-i-index-vectors-for-cosine-similarity
    faiss.normalize_L2(embedding)
    return embedding


def build_index():
    savedir = Path(settings.INDEX_SAVE_DIR)
    if savedir.exists():
        logger.info("index already exists @ %s, will not overwrite", savedir)
        return

    embedding_dim = llamafile.embed("Apples are red.").shape[-1]

    # index uses cosine similarity
    # see: https://github.com/facebookresearch/faiss/wiki/MetricType-and-distances#how-can-i-index-vectors-for-cosine-similarity
    index = faiss.IndexFlatIP(embedding_dim)

    docs = []
    for text in load_data_for_indexing():
        embedding = embed(text)
        index.add(embedding)
        docs.append(text)

    savedir.mkdir(parents=True)
    faiss.write_index(index, str(savedir / "index.faiss"))
    with open(savedir / "index.json", "w") as fout:
        json.dump(docs, fout)


def load_index():
    savedir = Path(settings.INDEX_SAVE_DIR)
    if not savedir.exists():
        raise FileNotFoundError(f"index not found @ {savedir}")

    index = faiss.read_index(str(savedir / "index.faiss"))
    logger.info("index with %d entries loaded from %s", index.ntotal, savedir)

    with open(savedir / "index.json", "r") as fin:
        docs = json.load(fin)
    return index, docs


def pprint_search_results(scores: np.ndarray, doc_indices: np.ndarray, docs: list[str]):
    print("=== Search Results ===")
    for i, doc_ix in enumerate(doc_indices[0]):
        print('%.4f - "%s"' % (scores[0, i], docs[doc_ix][:100]))
    print()


SEP = "-" * 80


def run_query(k: int, index: faiss.IndexFlatIP, docs: list[str]):
    query = click.prompt(
        text="Enter query (ctrl-d to quit):",
        prompt_suffix="> ",
        default="What does Alice like?",
        show_default=True,
    )

    print("=== Query ===")
    print(query)
    print()

    # Vector search for top-k most similar documents
    emb = embed(query)
    scores, doc_indices = index.search(emb, k)
    pprint_search_results(scores, doc_indices, docs)
    search_results = [docs[ix] for ix in doc_indices[0]]

    print("=== Prompt ===")
    prompt_template = (
        "You are an expert Q&A system. Answer the user's query using the provided context information.\n"
        "Context information:\n"
        "%s\n"
        "Query: %s"
    )
    prompt = prompt_template % ("\n".join(search_results), query)
    print(f'"{prompt}"')
    prompt_ntokens = len(llamafile.tokenize(prompt, port=settings.GENERATION_MODEL_PORT))
    print(f"(prompt_ntokens: {prompt_ntokens})")

    print()
    print()

    print("=== Answer ===")
    answer = llamafile.completion(prompt)
    print(f'"{answer}"')
    print()
    print(SEP)


@click.group(invoke_without_command=True)
@click.pass_context
def cli(ctx: click.Context):
    # Invoke `rag` by default
    if ctx.invoked_subcommand is None:
        ctx.invoke(rag)


@cli.command()
@click.option(
    "-k",
    "--k-search-results",
    default=3,
    help="Number of search results to add to the prompt.",
)
def rag(k_search_results: int):
    index, docs = load_index()
    while True:
        run_query(k_search_results, index, docs)


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    build_index()
    cli()
