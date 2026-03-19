"""
BM25 Handler Module
Centralizes BM25 functionality for ShopTalk project
"""

import os
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from rank_bm25 import BM25Okapi
from typing import Dict, List, Tuple, Any, Optional
import traceback

# Ensure NLTK resources are available
def ensure_nltk_resources():
    """Ensure all required NLTK resources are downloaded."""
    required_resources = [
        ('punkt', 'tokenizers/punkt'),
        ('stopwords', 'corpora/stopwords')
    ]
    
    for resource, path in required_resources:
        try:
            nltk.data.find(path)
        except LookupError:
            print(f"Downloading NLTK resource '{resource}'...")
            nltk.download(resource, quiet=True)

# Cache for BM25 models
_BM25_CACHE = {}  # Key: collection_name, Value: (bm25_model, corpus_mapping)

def get_bm25_model(collection_name: str, data_path: str) -> Tuple[Any, Dict]:
    """
    Load BM25 model for a collection if not already loaded.
    
    Args:
        collection_name: Name of the collection
        data_path: Path to data directory
        
    Returns:
        Tuple of (bm25_model, corpus_mapping)
    """
    global _BM25_CACHE
    
    # Return cached model if available
    if collection_name in _BM25_CACHE:
        print(f"Using cached BM25 model for collection '{collection_name}'")
        return _BM25_CACHE[collection_name]
    
    # Paths to BM25 files
    bm25_path = os.path.join(data_path, f"{collection_name}_bm25.pkl")
    corpus_map_path = os.path.join(data_path, f"{collection_name}_corpus_map.pkl")
    
    # Load BM25 model if files exist
    if os.path.exists(bm25_path) and os.path.exists(corpus_map_path):
        try:
            with open(bm25_path, 'rb') as f:
                bm25_model = pickle.load(f)
            with open(corpus_map_path, 'rb') as f:
                corpus_mapping = pickle.load(f)
            
            # Cache the loaded models
            _BM25_CACHE[collection_name] = (bm25_model, corpus_mapping)
            print(f"BM25 model for collection '{collection_name}' loaded and cached")
            return _BM25_CACHE[collection_name]
        except Exception as e:
            print(f"Error loading BM25 model for collection '{collection_name}': {e}")
    
    return None, None

def clear_bm25_cache(collection_name: str = None):
    """
    Clear BM25 model from cache.
    
    Args:
        collection_name: Specific collection to clear, or None to clear all
    """
    global _BM25_CACHE
    
    if collection_name:
        if collection_name in _BM25_CACHE:
            del _BM25_CACHE[collection_name]
            print(f"Cleared cached BM25 model for collection '{collection_name}'")
    else:
        _BM25_CACHE.clear()
        print("Cleared all cached BM25 models")

def create_bm25_index(collection_name: str, data_path: str, texts: List[str], 
                      product_ids: List[str]) -> bool:
    """
    Create and save BM25 index for a collection.
    
    Args:
        collection_name: Name of the collection
        data_path: Path to data directory
        texts: List of processed text documents
        product_ids: List of product IDs corresponding to texts
        
    Returns:
        Success status (bool)
    """
    ensure_nltk_resources()
    stop_words = set(stopwords.words('english'))
    
    try:
        print("Starting BM25 tokenization...")
        tokenized_corpus = []
        for text in texts:
            tokens = word_tokenize(text.lower())
            tokens = [t for t in tokens if t not in stop_words and len(t) > 2]
            tokenized_corpus.append(tokens)
        
        print(f"Creating BM25 model with {len(tokenized_corpus)} documents")
        bm25 = BM25Okapi(tokenized_corpus)
        print("BM25 model created successfully")
        
        # Save BM25 model and corpus mapping to disk for later use
        bm25_path = os.path.join(data_path, f"{collection_name}_bm25.pkl")
        corpus_map_path = os.path.join(data_path, f"{collection_name}_corpus_map.pkl")
        
        # Save mapping between corpus index and product ID
        corpus_map = {i: product_ids[i] for i in range(len(product_ids))}
        
        with open(bm25_path, 'wb') as f:
            pickle.dump(bm25, f)
        
        with open(corpus_map_path, 'wb') as f:
            pickle.dump(corpus_map, f)
        
        # Update cache
        _BM25_CACHE[collection_name] = (bm25, corpus_map)
        
        print(f"BM25 index saved to {bm25_path}")
        return True
    except Exception as e:
        print(f"Error during BM25 creation: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def search_bm25(collection_name: str, query: str, 
                bm25_model: Any, corpus_mapping: Dict, top_k: int = 10) -> Dict[str, float]:
    """
    Search using BM25 model.
    
    Args:
        collection_name: Name of the collection
        query: User query string
        bm25_model: BM25 model
        corpus_mapping: Mapping from corpus index to product ID
        top_k: Number of top results to return
        
    Returns:
        Dictionary mapping product_id to BM25 score
    """
    ensure_nltk_resources()
    results_dict = {}
    
    try:
        # Tokenize and preprocess query for BM25
        stop_words = set(stopwords.words('english'))
        
        # Preprocess query
        query_tokens = word_tokenize(query.lower())
        query_tokens = [t for t in query_tokens if t not in stop_words and len(t) > 2]
        
        # Get BM25 scores
        if query_tokens:
            bm25_scores = bm25_model.get_scores(query_tokens)
            
            # Get top BM25 results
            bm25_top_indices = sorted(range(len(bm25_scores)), 
                                     key=lambda i: bm25_scores[i], 
                                     reverse=True)[:top_k]
            
            # Map indices to product IDs
            for idx in bm25_top_indices:
                if idx in corpus_mapping:
                    product_id = corpus_mapping[idx]
                    results_dict[product_id] = bm25_scores[idx]
            
            print(f"BM25 found {len(results_dict)} matching products")
    except Exception as e:
        print(f"Error in BM25 search: {e}")
    
    return results_dict