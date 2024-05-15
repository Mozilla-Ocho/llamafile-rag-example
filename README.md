# Minimal Local RAG using llamafile

A very basic interactive CLI for indexing and querying documents using [llamafiles](https://github.com/Mozilla-Ocho/llamafile) for embeddings and text generation. Index is based on a [FAISS](https://github.com/facebookresearch/faiss) vector store. Default embedding model is [mxbai-embed-large-v1](https://huggingface.co/mixedbread-ai/mxbai-embed-large-v1) ([llamafile link](https://huggingface.co/Mozilla/mxbai-embed-large-v1-llamafile)) and text generation model is [mistral-7b-instruct-v0.2](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2) ([llamafile link](https://huggingface.co/jartine/Mistral-7B-Instruct-v0.2-llamafile)). (These can be changed by editing `setup.sh`.)

Setup:

```bash
cp .env.example .env
./setup.sh
```
This script will download llamafiles from HuggingFace and may take several minutes depending on your internet connection.

NOTE: setup script requires pyenv

## Quickstart with toy data

To start the app, run:

```bash
./app.sh
```

When you run the app, it will:

1. Start two llamafile servers on separate ports, one for the embedding model (port 8080) and one for the text generation model (port 8081). This might take ~40 seconds.
2. If it's the first time you're running the app, it will automatically ingest the contents of the files in the `toy_data/` directory into a vector store (the "index"). Contents of the `toy_data/` directory:

```
1.txt: Alice likes red squares.
2.txt: Bob likes blue circles.
3.txt: Chris likes blue triangles.
4.txt: David does not like green triangles.
5.txt: Mary does not like circles.
```

3. After that's done, it will start an interactive CLI that allows you to ask a model questions about the data in the index. The CLI should look like:

```bash
Enter query (ctrl-d to quit): [What does Alice like?]>
```

If you just hit Enter here, by default the query will be "What does Alice like?". The app output should look like:

```
=== Query ===
What does Alice like?

=== Search Results ===
0.7104 - " alice likes red squares ."
0.5229 - " bob likes blue circles ."
0.4088 - " chris likes blue triangles ."

=== Prompt ===
"You are an expert Q&A system. Answer the user's query using the provided context information.
Context information:
 alice likes red squares .
 bob likes blue circles .
 chris likes blue triangles .
Query: What does Alice like?"
(prompt_ntokens: 55)


=== Answer ===
"
Answer: Alice likes red squares."

--------------------------------------------------------------------------------
```

Here some other queries you could try:

* Who hates three-sided shapes?
* Who likes shapes that are the color of the sky?
* Who likes rectangles?

That's pretty much it.


## App Configuration

You can change most app settings via the `.env` file. The default file should look like:

```
EMBEDDING_MODEL_PORT=8080
GENERATION_MODEL_PORT=8081
INDEX_LOCAL_DATA_DIRS=local_data,toy_data
INDEX_TEXT_CHUNK_LEN=128
INDEX_SAVE_DIR=./index-toy
```

See [settings.py](settings.py) for all available options. 

### Using different models

By default, the app uses:

* Embeddings: 
* Text generation:


### Adding your own data

By default, the app is configured to index the contents of the directories listed in `INDEX_LOCAL_DATA_DIRS`, which are `local_data` and `toy_data`. Currently we only support indexing `.txt` files. 

First, in your `.env`, change `INDEX_SAVE_DIR` to wherever you want your index to be saved. The app will not change or overwrite an existing index, so either change the directory in the `.env` or delete the existing index at `./index-toy`.

There are 2 ways to add data:

1. Add `.txt` files to the `local_data/` directory. You can remove `toy_data/` from the `INDEX_LOCAL_DATA_DIRS` list in our `.env` file. You can also just add another directory to the `INDEX_LOCAL_DATA_DIRS` list.
2. Add web pages to the index by specifying one or more URLs to the `INDEX_URLS` var in your `.env` file, e.g. `INDEX_URLS=url1,url2,...`.


