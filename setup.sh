#!/bin/bash

# Create virtualenv
if [ ! -d "venv" ]; then
  pyenv local 3.11
  python3 -m venv venv
  source venv/bin/activate
  pip install --upgrade pip setuptools wheel
  pip install -r requirements.txt
fi

# Create .env to store app settings (see also settings.py)
if [ ! -f ".env" ]; then cp -v .env.example .env; fi

#
# Download llamafiles then symlink them to
# - models/embedding_model.llamafile
# - models/generation_model.llamafile
#
EMBEDDING_MODEL_URL="https://huggingface.co/Mozilla/mxbai-embed-large-v1-llamafile/resolve/main/mxbai-embed-large-v1-f16.llamafile"
GENERATION_MODEL_URL="https://huggingface.co/Mozilla/Mistral-7B-Instruct-v0.2-llamafile/resolve/main/mistral-7b-instruct-v0.2.Q4_0.llamafile"

function url_to_filename() {
  url=$1
  filename="${url##*/}"
  echo "${filename}"
}

mkdir -pv models
cd models || exit

if [ ! -f "embedding_model.llamafile" ]
then
  # Download and symlink embedding model
  wget -nc "${EMBEDDING_MODEL_URL}"
  filename="$(url_to_filename "${EMBEDDING_MODEL_URL}")"
  chmod +x "${filename}"
  ln -s "${filename}" embedding_model.llamafile
fi

if [ ! -f "generation_model.llamafile" ]
then
  # Download and symlink generation model
  wget -nc "${GENERATION_MODEL_URL}"
  filename="$(url_to_filename "${GENERATION_MODEL_URL}")"
  chmod +x "${filename}"
  ln -s "${filename}" generation_model.llamafile
fi

cd - || exit


