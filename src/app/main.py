#!/usr/bin/env python3

import json
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Iterator
from datetime import datetime
import time
import os
import sys

# LlamaIndex with Ollama
from llama_index.core import Document, VectorStoreIndex, Settings
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.storage.storage_context import StorageContext
from llama_index.core.node_parser import SentenceSplitter
import chromadb

# Setup paths relative to script
from dotenv import load_dotenv
load_dotenv(override=True)

# rescast all the sizes from .env into integer
temp = os.getenv("_BATCH_SIZE", 10)
_BATCH_SIZE = int(temp)

temp = os.getenv('_BUFFER_SIZE_PDF', 4096)
_BUFFER_SIZE_PDF = int(temp)

temp = os.getenv('_BUFFER_SIZE_TEXT', 10240)
_BUFFER_SIZE_TEXT = int(temp)

temp = os.getenv('_CHUNK_SIZE', 512)
_CHUNK_SIZE = int(temp)

temp = os.getenv('_CHUNK_OVERLAP', 50)
_CHUNK_OVERLAP = int(temp)

# PDF processing
try:    
    import PyPDF2
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False


def _read_pdf_stream(file_path: Path,
                     buffer_size: int = _BUFFER_SIZE_PDF) -> Iterator[str]:
    """Reads PDF content page by page, yielding text to manage memory."""
    if not PDF_SUPPORT:
        raise ImportError("PyPDF2 is required for PDF processing. Install with: pip install PyPDF2")
    
    with open(file_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        for page in pdf_reader.pages:
            yield page.extract_text() + "\n"


def _read_text_stream(file_path: Path,
                      buffer_size: int = _BUFFER_SIZE_TEXT) -> Iterator[str]:
    """Reads text file in chunks, yielding text to manage memory."""
    for encoding in ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']:
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                while True:
                    chunk = f.read(buffer_size)
                    if not chunk:
                        break
                    yield chunk
            return
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError(f"Could not decode {file_path} with any common encoding")


def read_and_chunk_file_generator(file_path: str,
                                  chunk_size: int = _CHUNK_SIZE,
                                  chunk_overlap: int = _CHUNK_OVERLAP,
                                  file_id: Optional[str] = None) -> Iterator[Document]:
    """
    Reads a large file in a memory-efficient way and yields LlamaIndex Documents.
    
    Args:
        file_path: Path to the input file (.txt or .pdf).
        chunk_size: Target size of each chunk.
        chunk_overlap: Overlap between chunks.
        file_id: Optional, a unique ID for the file.
        
    Yields:
        LlamaIndex Document objects.
    """
    file_path_obj = Path(file_path)
    if not file_path_obj.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    file_ext = file_path_obj.suffix.lower()
    if file_ext not in ('.txt', '.pdf'):
        raise ValueError(f"Unsupported file type: {file_ext}")
    
    # Create file ID if not provided
    if file_id is None:
        file_id = hashlib.md5(file_path.encode()).hexdigest()[:12]
    
    print(f"🔧 Starting streaming chunking for file_id: {file_id}")
    
    # Initialize the text splitter
    text_splitter = SentenceSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separator="\n\n"
    )
    
    # Use the appropriate reading stream based on file type
    stream = _read_pdf_stream(file_path_obj) if file_ext == '.pdf' else _read_text_stream(file_path_obj)
    
    buffer = ""
    chunk_index = 0
    for file_chunk in stream:
        buffer += file_chunk
        
        # Split the buffer. Splitter returns a list of chunks.
        chunks_from_buffer = text_splitter.split_text(buffer)
        
        # If there's more than one chunk, all but the last one are guaranteed to be complete.
        if len(chunks_from_buffer) > 1:
            for chunk_text in chunks_from_buffer[:-1]:
                metadata = {
                    'source': file_path_obj.name,
                    'file_id': file_id,
                    'chunk_index': chunk_index,
                    'content_length': len(chunk_text),
                    'file_type': file_ext,
                    'created_at': datetime.now().isoformat()
                }
                yield Document(text=chunk_text, metadata=metadata, id_=f"{file_id}_chunk_{chunk_index:06d}")
                chunk_index += 1
            
            # The last chunk of the split is the new buffer
            buffer = chunks_from_buffer[-1]
            
    # After the loop, process any remaining text in the buffer
    if buffer:
        metadata = {
            'source': file_path_obj.name,
            'file_id': file_id,
            'chunk_index': chunk_index,
            'content_length': len(buffer),
            'file_type': file_ext,
            'created_at': datetime.now().isoformat()
        }
        yield Document(text=buffer, metadata=metadata, id_=f"{file_id}_chunk_{chunk_index:06d}")

    print(f"✅ Streaming chunking complete. Total chunks yielded: {chunk_index + 1}")


def embed_chunks(input_file: str,
                 collection_name: str = "documents",
                 persist_directory: str = None,
                 ollama_model: str = "llama3.2:3b",
                 hf_embed_model: str = "BAAI/bge-small-en-v1.5",
                 batch_size: int = 10,
                 resume: bool = True) -> int:
    """
    Embeds a file's content directly into ChromaDB in a memory-efficient streaming manner.
    """
    input_path = Path(input_file)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")
    
    if persist_directory is None:
        persist_directory = str(_collections)
    
    # Setup Ollama LLM
    print("🚀 Initializing Ollama LLM model...")
    llm = Ollama(model=ollama_model, request_timeout=120.0)
    
    # Setup Hugging Face Embedding
    print(f"🚀 Initializing HuggingFace embedding model: {hf_embed_model}")
    embed_model = HuggingFaceEmbedding(model_name=hf_embed_model, device="cpu")
    
    # Set global settings
    Settings.llm = llm
    Settings.embed_model = embed_model
    print("✅ All models initialized successfully")
    
    # Setup ChromaDB
    print(f"🔧 Connecting to ChromaDB collection: {collection_name}")
    Path(persist_directory).mkdir(parents=True, exist_ok=True)
    chroma_client = chromadb.PersistentClient(path=persist_directory)
    
    try:
        collection = chroma_client.get_or_create_collection(name=collection_name)
        print(f"✓ Using existing or new collection: {collection_name}")
    except Exception as e:
        print(f"❌ Error setting up ChromaDB: {e}")
        raise
    
    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    
    # --- The streaming embedding loop ---
    embedded_count = 0
    batch_docs = []
    
    # Get a generator for the chunks
    chunk_generator = read_and_chunk_file_generator(str(input_path))
    
    print(f"Starting to stream and embed in batches of {batch_size}...")
    
    for doc in chunk_generator:
        # Check if this document already exists in ChromaDB (for resuming)
        # NOTE: This check might be slow for large collections.
        if resume:
            results = collection.get(ids=[doc.doc_id])
            if results['ids']:
                print(f"⏭️ Skipping existing chunk: {doc.doc_id}")
                continue
        
        batch_docs.append(doc)
        
        if len(batch_docs) >= batch_size:
            print(f"🔄 Embedding batch of {len(batch_docs)} chunks...")
            index = VectorStoreIndex.from_documents(
                batch_docs,
                storage_context=storage_context,
                embed_model=embed_model,
                transformations=[],
                show_progress=True
            )
            embedded_count += len(batch_docs)
            batch_docs = []
            
    # Embed any remaining documents in the final batch
    if batch_docs:
        print(f"🔄 Embedding final batch of {len(batch_docs)} chunks...")
        index = VectorStoreIndex.from_documents(
            batch_docs,
            storage_context=storage_context,
            embed_model=embed_model,
            transformations=[],
            show_progress=True
        )
        embedded_count += len(batch_docs)
    
    print(f"✅ Embedding complete. Total chunks embedded: {embedded_count}")
    return embedded_count


def main():
    print(os.getenv('_LOC_RAW'))
    input_file: str = None
    collection_name: str = None

    if len(sys.argv) == 3:
        input_file: str = f"{os.getenv('_LOC_RAW')}/{sys.argv[1]}"
        collection_name: str = sys.argv[2]

    if input_file and collection_name:
        print(f"Processing file: {input_file}, embeddings will be stored in collection {collection_name}")    
        embedded = embed_chunks(
            input_file=input_file,
            collection_name=collection_name,
            persist_directory=os.getenv('_LOC_PROCESSED'),
            ollama_model=os.getenv('_LLM_MODEL', "llama3.2:3b"),
            hf_embed_model=os.getenv('_EMBED_MODEL', "BAAI/bge-small-en-v1.5"),
            batch_size=_BATCH_SIZE,
            resume=True
        )
        print(f"Embedded chunks stored: {embedded}")
        print("\n✅ All done!")
    else:
        print(f"Usage: main <input_file> <collection_name>")
        print(f"input file assumed in subdir data/raw")
        print(f"Collection will be created in subdir data/processed")


# Example usage:
if __name__ == "__main__":
    main()
