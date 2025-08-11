### Rag Guru
---
A lightweight chunking and embedding engine.

This engine provides a simple CLI to chunk and embed text and pdf files using a streaming generator to accomodate for large files.

1. Ollama: For accessing LLMs & Embedding models
2. Huggingface: For embedding API
3. ChromaDB: For storing the embeddings

The .env has several defaults that will be used to influence the chunking and embedding process.

usage:
    main.py:
     - input_file (.txt or .pdf)
     - collection_name (for ChromaDB)

