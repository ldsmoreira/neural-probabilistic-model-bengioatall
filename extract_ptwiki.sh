#!/bin/bash

# Configuration
WIKI_DUMP_URL="https://dumps.wikimedia.org/ptwiki/latest/ptwiki-latest-pages-articles.xml.bz2"
DOWNLOAD_DIR="data/raw"
EXTRACTED_DIR="${DOWNLOAD_DIR}/ptwiki-latest-pages-articles"

# Create directories
mkdir -p "${DOWNLOAD_DIR}"
mkdir -p "${EXTRACTED_DIR}"

# Step 1: Download the Wikipedia dump
DUMP_FILE_PATH="${DOWNLOAD_DIR}/ptwiki-latest-pages-articles.xml.bz2"
if [ ! -f "${DUMP_FILE_PATH}" ]; then
    echo "Downloading Wikipedia dump from ${WIKI_DUMP_URL}..."
    wget -O "${DUMP_FILE_PATH}" "${WIKI_DUMP_URL}"
else
    echo "Dump file already exists at ${DUMP_FILE_PATH}."
fi

# Step 2: Extract the Wikipedia dump
echo "Extracting Wikipedia dump using WikiExtractor..."
wikiextractor "${DUMP_FILE_PATH}" -o "${EXTRACTED_DIR}"

echo "Extraction complete. Extracted data is available at ${EXTRACTED_DIR}."
