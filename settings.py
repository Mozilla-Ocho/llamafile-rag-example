import os
from dotenv import load_dotenv
load_dotenv(dotenv_path=".env")

####
# llamafile server settings
####
EMBEDDING_MODEL_PORT: int = int(os.getenv("EMBEDDING_MODEL_PORT", "8080"))
GENERATION_MODEL_PORT: int = int(os.getenv("GENERATION_MODEL_PORT", "8081"))

####
# Indexing settings
####
INDEX_URLS: list[str] = os.getenv("INDEX_URLS").split(",") if os.getenv("INDEX_URLS") else []
INDEX_LOCAL_DATA_DIRS: list[str] = os.getenv("INDEX_LOCAL_DATA_DIRS", "./local_data").split(",")

# documents will be split into snippets of size <chunk_len> before indexing
# if set to -1, will using `EMBEDDING_MODEL_MAX_LEN`
INDEX_TEXT_CHUNK_LEN: int = int(os.getenv("INDEX_TEXT_CHUNK_LEN", -1))

# index will be saved to this directory
INDEX_SAVE_DIR: str = os.getenv("INDEX_SAVE_DIR", "./index")

####
# Other model/text processing settings
####
# TODO: this should be set using the model config
EMBEDDING_MODEL_MAX_LEN: int = int(os.getenv("EMBEDDING_MODEL_MAX_LEN", "512"))
# GENERATION_MODEL_MAX_LEN: int = int(os.getenv("GENERATION_MODEL_MAX_LEN", "512"))
