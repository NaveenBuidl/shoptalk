"""
Enhanced ShopTalk Implementation
Optimized based on actual dataset analysis and common query patterns
"""

print("rag.py is loaded by FastAPI")

#  IMPORTS AND CONFIGURATION
# Standard library imports
import os
import gc
import re
import json
from openai import OpenAI
# from openai import ChatCompletion
from typing import List, Dict, Any


# FastAPI related imports
from fastapi import HTTPException, UploadFile, File, Depends
from pydantic import BaseModel, Field
from fastapi import Depends, HTTPException, APIRouter

# Data processing libraries
import pandas as pd
import numpy as np
from langchain_text_splitters import RecursiveCharacterTextSplitter
from tqdm import tqdm

# NLP and ML libraries
from sentence_transformers import SentenceTransformer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from rank_bm25 import BM25Okapi
from transformers import pipeline

# Vector database
from pymilvus import Collection, CollectionSchema, FieldSchema, DataType, utility
from pymilvus import connections

# Import configuration from separate modules
from config_embedding import EMBEDDING_MODEL_NAME, EMBEDDING_DIMENSION, SIMILARITY_METRIC
from config_fields import (
    FIELDS_CONFIG, 
    ID_COLUMNS, 
    EMBEDDING_PRIORITY_COLUMNS, 
    ALL_FIELDS, 
    ORIGINAL_TO_NEW_MAPPING, 
    NEW_TO_ORIGINAL_MAPPING
)
# Import BM25 handler module
from bm25_handler import (
    get_bm25_model,
    clear_bm25_cache,
    create_bm25_index,
    search_bm25
)

# Suppress warnings
import warnings
warnings.filterwarnings("ignore")

# openai api code
from dotenv import load_dotenv
load_dotenv()  # will look for `.env` in current or parent directories
# Set your key securely
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)
print("Loaded API key:", OPENAI_API_KEY[:6], "..." if OPENAI_API_KEY else "NOT FOUND")

# region GLOBAL VARIABLES AND INITIALIZATION
# Set up the FastAPI router
routerRag = APIRouter()

# Define paths
DATA_PATH = os.path.join(os.getcwd(), "data")
os.makedirs(DATA_PATH, exist_ok=True)  # Ensure the directory exists

# try:
#     # Initialize text generation model (for responding to queries)
#     # Using T5 for better text generation quality
#     llm_model_name = "google/flan-t5-small"
#     # llm_model_name = "google/flan-t5-base"
#     # llm_model_name = "google/flan-t5-large"
#     # llm_ model_name = "model="sshleifer/distilbart-cnn-6-6"
    
#     llm = pipeline("text2text-generation", model=llm_model_name)
    
#     print(f"Loaded {llm_model_name} model")
# except Exception as e:
#     print(f"Error loading {llm_model_name}: {e}")


# Initialize embedding model - using BGE for better semantic search performance
# EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"
# EMBEDDING_DIMENSION = 384
# SIMILARITY_METRIC = "cosine"
print(f"Using embedding model: {EMBEDDING_MODEL_NAME} with dim {EMBEDDING_DIMENSION}")
embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

# Field name mapping for backward compatibility
FIELD_NAME_MAPPING = ORIGINAL_TO_NEW_MAPPING
REVERSE_FIELD_MAPPING = NEW_TO_ORIGINAL_MAPPING
# All fields for collection schema
ALL_FIELDS = list(FIELDS_CONFIG.keys())

# endregion

# HELPER FUNCTIONS

# # Testing the LLM
# def test_llm():
#     """Test if the LLM is working properly"""
#     test_prompt = "Write a short description of a blue shirt."
#     print(f"Test prompt: {test_prompt}")
    
#     try:
#         response = llm(test_prompt, max_length=50)
#         print(f"LLM test response: {response}")
#         return response
#     except Exception as e:
#         print(f"LLM test error: {e}")
#         return None
# UTILITY FUNCTIONS
def ensure_nltk_resources():
    """
    Ensure all required NLTK resources are downloaded.
    This centralized function should be called before any NLTK operations.
    """
    import nltk
    
    # Define resources needed throughout the application
    required_resources = [
        ('punkt', 'tokenizers/punkt'),
        ('stopwords', 'corpora/stopwords')
    ]
    
    # Check and download each resource if not already present
    for resource, path in required_resources:
        try:
            nltk.data.find(path)
            print(f"NLTK resource '{resource}' is already available.")
        except LookupError:
            print(f"Downloading NLTK resource '{resource}'...")
            nltk.download(resource, quiet=True)
            print(f"NLTK resource '{resource}' has been downloaded.")

# Initialize NLTK resources
ensure_nltk_resources()
print("NLTK resources initialized")

# # Call this after model initialization
# test_result = test_llm()
# print("LLM test completed")

#  DATA PREPROCESSING AND VALIDATION
# Field Validation
def validate_field_length(value, max_length):
    """Truncate string values that exceed the maximum field length"""
    if value is None:
        return ""
    str_value = str(value)
    if len(str_value) > max_length:
        print(f"Truncating field value from {len(str_value)} to {max_length} characters")
        return str_value[:max_length]
    return str_value

# Convert old field names to new field names
def convert_field_names(df):
    """Convert dataframe columns from old field names to new field names"""
    renamed_columns = {}
    for old_name, new_name in FIELD_NAME_MAPPING.items():
        if old_name in df.columns:
            renamed_columns[old_name] = new_name
    
    # Rename only columns that exist in the dataframe
    if renamed_columns:
        df = df.rename(columns=renamed_columns)
    
    return df

# EMBEDDING GENERATION
# Text embedding
def generate_embeddings(text_list, batch_size=32):
    """Generate embeddings using Sentence Transformer."""
    # Handle empty text
    cleaned_texts = [text if text and str(text).lower() != "nan" else "empty" for text in text_list]
    
    # Generate embeddings with sentence-transformers
    embeddings = embedding_model.encode(
        cleaned_texts, 
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True
    )
    
    return embeddings

# Enhanced document structure - Optimized for actual product distribution
def create_structured_document(product):
    """
    Create structured document representation optimized for actual product distribution.
    Uses field ordering from config_fields.py but new field names.
    """
    parts = []
    
    # Iterate through fields in priority order using config
    for field in EMBEDDING_PRIORITY_COLUMNS:
        # Try to get value using both new and original field names
        field_value = None
        if field in product:
            field_value = product[field]
        elif field in REVERSE_FIELD_MAPPING and REVERSE_FIELD_MAPPING[field] in product:
            field_value = product[REVERSE_FIELD_MAPPING[field]]
            
        # Skip empty values
        if not field_value or str(field_value).lower() == 'nan':
            continue
            
        # Format based on field type
        if field == 'product_title':
            parts.append(f"Title: {field_value} {field_value}")
        elif field == 'product_category':
            parts.append(f"Category: {field_value} {field_value} {field_value}")
        elif field == 'search_terms':
            # Truncate keywords to avoid dominating the embedding
            truncated_text = str(field_value)
            if len(truncated_text) > 1000:
                truncated_text = truncated_text[:1000]
            parts.append(f"Keywords: {truncated_text}")
        elif field == 'product_features':
            # Truncate features if too long
            features_text = str(field_value)
            if len(features_text) > 1500:
                features_text = features_text[:1500]
            parts.append(f"Features: {features_text}")
        elif field == 'visual_description':
            if field_value != "[undetermined product]":
                parts.append(f"Visual: {field_value}")
        elif field == 'manufacturer_brand':
            parts.append(f"Brand: {field_value}")
        elif field == 'primary_color':
            parts.append(f"Color: {field_value}")
        elif field == 'primary_material':
            parts.append(f"Material: {field_value}")
        elif field == 'design_style':
            parts.append(f"Style: {field_value}")
        else:
            # Generic field handling
            field_name = field.replace('_', ' ').title()
            parts.append(f"{field_name}: {field_value}")
    
    return " ".join(parts)

# QUERY UNDERSTANDING - Aligned with actual product distribution
# Uses a consolidated query processing pipeline that extracts intent and 
# creates structured queries in a single pass for improved consistency
def process_query(query):
    """
    Comprehensive query processing function that:
    1. Extracts semantic intent and attributes
    2. Creates a structured representation for vector search that aligns with document structure
    3. Returns both intent and structured query for downstream use
    
    Args:
        query (str): The original user query
    
    Returns:
        tuple: (structured_query, intent) where:
            - structured_query is optimized for vector search
            - intent is a dictionary of extracted semantic information
    """
    query_lower = query.lower()
    
    # Initialize intent structure
    intent = {
        "category": None,
        "attributes": [],
        "material": None,
        "style": None,
        "brand": None,
        "functionality": []
    }
    
    # Category mapping based on actual product distribution
    categories = {
        # Phone accessories
        "phone case": "CELLULAR_PHONE_CASE", 
        "phone cover": "CELLULAR_PHONE_CASE",
        "cell case": "CELLULAR_PHONE_CASE",
        "mobile case": "CELLULAR_PHONE_CASE",
        "cellphone case": "CELLULAR_PHONE_CASE",
        "cellular case": "CELLULAR_PHONE_CASE",
        "iphone case": "CELLULAR_PHONE_CASE",
        "samsung case": "CELLULAR_PHONE_CASE",
        "phone": "CELLULAR_PHONE",
        "smartphone": "CELLULAR_PHONE",
        "mobile phone": "CELLULAR_PHONE",
        
        # Shoes/Footwear
        "shoe": "SHOES", 
        "footwear": "SHOES", 
        "sneaker": "SHOES", 
        "boot": "BOOT",
        "boots": "BOOT",
        "sandal": "SANDAL",
        "sandals": "SANDAL",
        "slipper": "SHOES",
        "slippers": "SHOES",
        "heel": "SHOES",
        "heels": "SHOES",
        "loafer": "SHOES",
        "running shoe": "SHOES",
        "walking shoe": "SHOES",
        # Add men's shoes/ women's shoes
        
        # Grocery
        "food": "GROCERY", 
        "snack": "GROCERY",
        "grocery": "GROCERY",
        "meal": "GROCERY",
        "fruit": "GROCERY",
        "vegetable": "GROCERY",
        "drink": "GROCERY",
        "beverage": "GROCERY",
        # icecream?
        
        # Books
        "book": "BOOK", 
        "novel": "BOOK",
        "textbook": "BOOK",
        "paperback": "BOOK",
        "hardcover": "BOOK",
        "ebook": "BOOK",
        
        # Furniture
        "chair": "CHAIR",
        "sofa": "SOFA", 
        "couch": "SOFA", 
        "table": "TABLE",
        "desk": "TABLE",
        "bed": "BED",
        "dresser": "DRESSER",
        "nightstand": "NIGHTSTAND",
        "furniture": "HOME_FURNITURE",
        "shelf": "SHELF",
        "cabinet": "CABINET",
        
        # Pet supplies
        "pet bed": "PET_SUPPLIES",
        "dog bed": "PET_SUPPLIES",
        "cat bed": "PET_SUPPLIES",
        "pet food": "PET_SUPPLIES",
        "pet toy": "PET_SUPPLIES",
        "pet supplies": "PET_SUPPLIES",
        
        # Electronics
        "laptop": "COMPUTER",
        "computer": "COMPUTER",
        "monitor": "MONITOR",
        "television": "TELEVISION",
        "tv": "TELEVISION",
        "headphone": "HEADPHONES",
        "headphones": "HEADPHONES",
        "speaker": "SPEAKER",
        "camera": "CAMERA"
        
        # Add jewellery/earrings/necklace
    }
    
    # Common materials in queries
    materials = ["leather", "wood", "wooden", "cotton", "plastic", "metal", "glass", 
                "stainless steel", "fabric", "polyester", "aluminum", "ceramic"]
    
    # Common styles in queries
    styles = ["modern", "traditional", "rustic", "minimalist", "contemporary", 
             "industrial", "vintage", "scandinavian", "bohemian", "mid-century"]
    
    # Common brands 
    brands = ["amazon", "stone & beam", "rivet", "amazonbasics", "nike", "apple", 
             "samsung", "adidas", "puma", "sony"]
    
    # Functionality aspects
    functionalities = ["waterproof", "wireless", "bluetooth", "adjustable", "foldable", 
                     "portable", "rechargeable", "ergonomic", "extendable", "reclining"]
    
    # Extract category with priority for multi-word categories
    for term in ["phone case", "cell case", "phone cover", "mobile case"]:
        if term in query_lower:
            intent["category"] = "CELLULAR_PHONE_CASE"
            break
    
    # If no multi-word match, try single words
    if not intent["category"]:
        for term, category in categories.items():
            if term in query_lower:
                intent["category"] = category
                break
    
    # Extract material
    for material in materials:
        if material in query_lower:
            intent["material"] = material
            break
    
    # Extract style
    for style in styles:
        if style in query_lower:
            intent["style"] = style
            break
    
    # Extract brand
    for brand in brands:
        if brand in query_lower:
            intent["brand"] = brand
            break
    
    # Extract functionality
    for func in functionalities:
        if func in query_lower:
            intent["functionality"].append(func)
    
    # Extract attributes (colors, sizes, etc.)
    colors = ["red", "blue", "green", "black", "white", "yellow", "purple", 
             "pink", "brown", "gray", "grey", "orange", "beige"]
    
    sizes = ["small", "medium", "large", "extra large", "xl", "xxl", 
            "king", "queen", "twin", "full"]
    
    # Add colors to attributes
    for color in colors:
        if color in query_lower:
            intent["attributes"].append({"type": "color", "value": color})
    
    # Add sizes to attributes
    for size in sizes:
        if size in query_lower:
            intent["attributes"].append({"type": "size", "value": size})
    
    # Now create the structured query based on extracted intent
    # This is where we align with create_structured_document() structure
    structured_parts = []
    
    # Start with the original query as a title-like element
    structured_parts.append(f"Title: {query} {query}")
    
    # Emphasize category if detected (triple repetition like in create_structured_document)
    if intent["category"]:
        structured_parts.append(f"Category: {intent['category']} {intent['category']} {intent['category']}")
    
    # Add keywords based on extracted intent
    keywords = []
    if intent["material"]:
        keywords.append(intent["material"])
    if intent["style"]:
        keywords.append(intent["style"])
    if intent["brand"]:
        keywords.append(intent["brand"])
    for attr in intent["attributes"]:
        keywords.append(attr["value"])
    for func in intent["functionality"]:
        keywords.append(func)
    
    if keywords:
        # Limit keywords to 1000 chars like in create_structured_document
        keyword_text = " ".join(keywords)
        if len(keyword_text) > 1000:
            keyword_text = keyword_text[:1000]
        structured_parts.append(f"Keywords: {keyword_text}")
    
    # Add material as a dedicated field (matching document structure)
    if intent["material"]:
        structured_parts.append(f"Material: {intent['material']}")
    
    # Add color as a dedicated field
    color_attrs = [attr for attr in intent["attributes"] if attr["type"] == "color"]
    if color_attrs:
        color_values = [attr["value"] for attr in color_attrs]
        structured_parts.append(f"Color: {' '.join(color_values)}")
    
    # Add brand as a dedicated field
    if intent["brand"]:
        structured_parts.append(f"Brand: {intent['brand']}")
    
    # Add style as a dedicated field
    if intent["style"]:
        structured_parts.append(f"Style: {intent['style']}")
    
    # Add functionality as features (matching document structure)
    if intent["functionality"]:
        functionality_text = " ".join(intent["functionality"])
        structured_parts.append(f"Features: {functionality_text}")
    
    # Create final structured query
    structured_query = " ".join(structured_parts)
    
    return structured_query, intent

# SEARCH AND RERANKING
def rank_documents(query, candidate_docs, bm25_results_dict=None, top_k=3):
    """
    Unified document ranking function that combines vector similarity, 
    text matching, BM25 scores, and intent matching into a single relevance score.
    
    Args:
        query (str): The original user query
        candidate_docs (list): List of document dictionaries with _vector_score field
        bm25_results_dict (dict, optional): Dictionary mapping product_id to BM25 scores
        top_k (int): Number of top results to return
        
    Returns:
        list: Top k documents sorted by relevance
    """
    # Extract query intent
    _, intent = process_query(query)
    bm25_results_dict = bm25_results_dict or {}
    
    # Score each result with component scoring functions
    ranked_docs = []
    for doc in candidate_docs:
        # Initialize scoring components
        scores = {
            "vector": 0.0,
            "text_match": 0.0,
            "bm25": 0.0,
            "intent_match": 0.0
        }
        
        # Get product ID for BM25 lookup
        product_id = doc.get("product_id", "unknown_id")
        
        # 1. Vector similarity score component
        scores["vector"] = _calculate_vector_score(doc)
        
        # 2. Text match score component
        scores["text_match"] = _calculate_text_match_score(query, doc)
        
        # 3. BM25 score component
        scores["bm25"] = _calculate_bm25_score(product_id, bm25_results_dict)
        
        # 4. Intent match score component (category, attributes, material, style, brand)
        scores["intent_match"] = _calculate_intent_match_score(intent, doc)
        
        # 5. Calculate final weighted score
        final_score = _calculate_final_score(scores)
        
        # Store document with its score
        ranked_docs.append((doc, final_score))

    # Sort by final score (descending)
    ranked_docs.sort(key=lambda x: x[1], reverse=True)
    
    # Return top k filtered results
    return [item[0] for item in ranked_docs[:top_k]]

def _calculate_vector_score(doc):
    """Extract and normalize vector similarity score"""
    # Get vector score from the document
    vector_score = doc.get("_vector_score", 0)
    return vector_score  # Already normalized between 0-1

def _calculate_text_match_score(query, doc, field_weights=None):
    """
    Calculate a text match score based on exact keyword matches in different fields.
    
    Args:
        query (str): The user query
        doc (dict): Document with fields to search in
        field_weights (dict): Weights for different fields (optional)
    
    Returns:
        float: Match score
    """
    if field_weights is None:
        field_weights = {
            "product_title": 3.0,      # Title matches are most important
            "product_category": 2.5,    # Category matches are very important
            "manufacturer_brand": 2.0,  # Brand matches are important
            "product_features": 1.5,    # Feature matches are moderately important
            "primary_color": 1.8,       # Color matches are important
            "primary_material": 1.7,    # Material matches are important
            "design_style": 1.6,        # Style matches are important
        }
    
    # Normalize and clean query
    query_lower = query.lower()
    query_terms = query_lower.split()
    
    # Calculate score
    score = 0.0
    
    # Check for exact field matches with field weighting
    for field, weight in field_weights.items():
        if field in doc and doc[field]:
            field_text = str(doc[field]).lower()
            
            # Exact field match (full query appears in this field)
            if query_lower in field_text:
                score += weight * 2.0
            
            # Individual term matches
            term_matches = sum(1 for term in query_terms if term in field_text)
            if term_matches > 0:
                score += weight * (term_matches / len(query_terms))
            
            # Bonus for terms appearing close together (phrase-like matches)
            for i in range(len(query_terms) - 1):
                if query_terms[i] in field_text and query_terms[i+1] in field_text:
                    # Check if terms appear close together
                    if f"{query_terms[i]} {query_terms[i+1]}" in field_text:
                        score += weight * 0.5
    
    # Normalize score to 0-1 range
    # If the maximum possible score is too hard to calculate, use a reasonable upper bound
    max_possible_score = sum(weight * 3.5 for weight in field_weights.values())
    normalized_score = min(1.0, score / max_possible_score) if max_possible_score > 0 else 0
    
    return normalized_score

def _calculate_bm25_score(product_id, bm25_results_dict):
    """Calculate normalized BM25 score for the document"""
    if product_id in bm25_results_dict:
        # Normalize BM25 score (they can be very large)
        raw_bm25_score = bm25_results_dict[product_id]
        bm25_score = min(1.0, raw_bm25_score / 10.0)  # Simple normalization
        return bm25_score
    return 0.0

def _calculate_intent_match_score(intent, doc):
    """
    Calculate how well the document matches the query intent
    (category, attributes, material, style, brand, functionality)
    
    Args:
        intent (dict): Query intent dictionary from process_query
        doc (dict): Document to check
        
    Returns:
        float: Intent match score (0-1)
    """
    intent_score = 0.0
    
    # Category match (most important)
    if intent["category"] and doc.get("product_category") == intent["category"]:
        intent_score += 1.0  # Major boost for exact category match
    # Related categories in tech accessories
    elif intent["category"] in ["CELLULAR_PHONE_CASE", "CELLULAR_PHONE"] and doc.get("product_category") in ["CELLULAR_PHONE_CASE", "CELLULAR_PHONE", "ACCESSORY"]:
        intent_score += 0.3  # Partial boost for related tech categories
    # Related categories in footwear
    elif intent["category"] in ["SHOES", "BOOT", "SANDAL"] and doc.get("product_category") in ["SHOES", "BOOT", "SANDAL"]:
        intent_score += 0.3  # Partial boost for related footwear
    
    # Attribute matches (especially color)
    for attr in intent["attributes"]:
        if attr["type"] == "color" and doc.get("primary_color") and attr["value"].lower() in doc.get("primary_color", "").lower():
            intent_score += 0.5
            
    # Material match
    if intent["material"] and doc.get("primary_material"):
        if intent["material"].lower() in doc.get("primary_material", "").lower():
            intent_score += 0.35  # Boost for material match
    
    # Style match
    if intent["style"] and (doc.get("design_style") or doc.get("visual_description")):
        style_text = f"{doc.get('design_style', '')} {doc.get('visual_description', '')}"
        if intent["style"].lower() in style_text.lower():
            intent_score += 0.35  # Boost for style match
    
    # Brand match
    if intent["brand"] and doc.get("manufacturer_brand"):
        if intent["brand"].lower() in doc.get("manufacturer_brand", "").lower():
            intent_score += 0.3  # Boost for brand match
    
    # Functionality match
    if intent["functionality"] and doc.get("product_features"):
        for func in intent["functionality"]:
            if func.lower() in doc.get("product_features", "").lower():
                intent_score += 0.25  # Boost for each functionality match
    
    # Normalize the score to 0-1 range
    # Assuming maximum possible score based on the above logic
    max_possible_score = 2.5  # Approximate maximum based on the boosts
    normalized_score = min(1.0, intent_score / max_possible_score)
    
    return normalized_score

def _calculate_final_score(scores):
    """
    Calculate final score by combining component scores with weights
    
    Args:
        scores (dict): Dictionary of component scores
        
    Returns:
        float: Final weighted score
    """
    # Component weights (sum to 1.0)
    weights = {
        "vector": 0.4,       # Vector similarity is important but not everything
        "text_match": 0.3,   # Text matching for exact matches
        "bm25": 0.1,         # BM25 for traditional keyword matching
        "intent_match": 0.2  # Intent matching for semantic understanding
    }
    
    # Calculate weighted sum
    final_score = sum(scores[component] * weight for component, weight in weights.items())
    
    return final_score

# DOCUMENT PROCESSING
# Main Processing Function
def process_documents(collection_name: str, file_name: str = None):
    """
    Process product data from CSV files and insert into Milvus.
    Creates structured embeddings with field prefixes following the priority order.
    Also creates BM25 index for hybrid search.
    """
    print("process_documents() has been triggered")
        
    # Create or get the collection
    collection = create_milvus_collection(collection_name)

    # Use specific file if provided, otherwise find first CSV
    if file_name:
        csv_path = os.path.join(DATA_PATH, file_name)
        if not os.path.exists(csv_path):
            print(f"File {file_name} not found in data directory.")
            return
    else:
        # Find CSV files in the data directory
        csv_files = [f for f in os.listdir(DATA_PATH) if f.endswith(".csv")]
        if not csv_files:
            print("No CSV files found in data directory.")
            return {"message": "No CSV found."}
        csv_path = os.path.join(DATA_PATH, csv_files[0])
    
    print(f"Loading data from: {csv_path}")
    
    # Load the CSV data
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} products from CSV.")
    
    # Convert old field names to new field names
    df = convert_field_names(df)
    
    # Limit to a smaller number of rows
    max_rows = 500  # Adjust based on your system
    print(f"Original dataset size: {len(df)} rows, max_rows is set to {max_rows}")

    # Force sampling regardless of size
    if len(df) > max_rows:
        print("Sampling data...")
        df = df.sample(n=min(max_rows, len(df)), random_state=42)
        print(f"After sampling: {len(df)} products selected for processing")
    
    # Load all fields into a dictionary for easier handling
    field_data = {}
    
    # Load all fields from the dataframe
    for field in ALL_FIELDS:
        field_data[field] = df[field].astype(str).tolist() if field in df.columns else [""] * len(df)
    
    # Create structured text for embedding using the enhanced document structure
    print("Creating structured text for embeddings...")
    structured_texts = []
    
    # Prepare data for BM25
    bm25_texts = []
    product_ids = field_data['product_id']
    
    for i in range(len(field_data['product_id'])):
        # Create a temporary product dictionary for this item
        product = {field: field_data[field][i] for field in ALL_FIELDS}
        
        # Use the enhanced document structure for vector search
        structured_text = create_structured_document(product)
        structured_texts.append(structured_text)
        
        # Create a plain text representation for BM25
        bm25_parts = []
        # Add fields with appropriate weighting
        if product.get('product_category'):
            bm25_parts.append(f"{product['product_category']} {product['product_category']} {product['product_category']}")
        if product.get('product_title'):
            bm25_parts.append(f"{product['product_title']} {product['product_title']}")
        if product.get('manufacturer_brand'):
            bm25_parts.append(f"{product['manufacturer_brand']} {product['manufacturer_brand']}")
        if product.get('primary_color'):
            bm25_parts.append(f"{product['primary_color']} {product['primary_color']}")
        if product.get('product_features') and str(product['product_features']).lower() != 'nan':
            bm25_parts.append(str(product['product_features'])[:300])  # Truncate features
            
        bm25_texts.append(" ".join(bm25_parts))
        
        # Print sample of the first item to verify
        if i == 0:
            print(f"Sample structured text: {structured_text[:200]}...")
            print(f"Sample BM25 text: {bm25_texts[0][:200]}...")
    
    # Create BM25 index with the handler
    print("Creating BM25 index...")
    success = create_bm25_index(
        collection_name=collection_name,
        data_path=DATA_PATH,
        texts=bm25_texts,
        product_ids=product_ids
    )
    if not success:
        print("Continuing without BM25 index...")
    
    # Generate embeddings for all products
    print("Generating embeddings...")
    embeddings = generate_embeddings(structured_texts, batch_size=32)
    
    # Clear memory
    print("Clearing memory...")
    del structured_texts
    gc.collect()
    
    # Add validation step to prevent max length errors
    print("Validating field lengths before insertion...")
    for field in ALL_FIELDS:
        if field in FIELDS_CONFIG:
            max_length = FIELDS_CONFIG[field].get('max_length', 100)
            field_data[field] = [validate_field_length(value, max_length) for value in field_data[field]]
        
    # Prepare data for insertion according to schema order
    data = []
    
    # Add all fields in schema order
    for field in ALL_FIELDS:
        data.append(field_data[field])
    
    # Add embeddings last
    data.append(embeddings)
    
    # Insert data into Milvus
    print(f"Inserting {len(field_data['product_id'])} products into Milvus...")
    collection.insert(data)
    print("Data inserted successfully.")
    
    # Create index for vector search
    print("Creating index on embedding field...")
    index_params = {
        "metric_type": SIMILARITY_METRIC,  
        "index_type": "IVF_FLAT",
        "params": {"nlist": 128}
    }
    collection.create_index(field_name="embedding", index_params=index_params)
    print("Index created successfully.")
    
    # Load collection for searching
    collection.load()
    print(f"Collection '{collection_name}' loaded and ready for queries.")

# Collection Creation
def create_milvus_collection(collection_name: str):
    """Creates or gets a Milvus collection with proper schema for product data."""
    fields = []
    
    # Add all fields according to the config
    for field_name in ALL_FIELDS:
        props = FIELDS_CONFIG[field_name]
        
        if props.get('is_primary', False):
            fields.append(FieldSchema(
                name=field_name,
                dtype=DataType.VARCHAR,
                max_length=props['max_length'],
                is_primary=True
            ))
        else:
            fields.append(FieldSchema(
                name=field_name,
                dtype=DataType.VARCHAR,
                max_length=props['max_length']
            ))
    
    # Add embedding field last
    fields.append(FieldSchema(
        name="embedding",
        dtype=DataType.FLOAT_VECTOR,
        dim=EMBEDDING_DIMENSION
    ))
    
    schema = CollectionSchema(fields, "Product collection with embeddings ordered by priority")

    # Drop collection if it exists with a different schema
    if utility.has_collection(collection_name):
        print(f"Collection '{collection_name}' already exists. Dropping...")
        utility.drop_collection(collection_name)

    # Create new collection
    collection = Collection(name=collection_name, schema=schema)
    print(f"Collection '{collection_name}' created.")
    
    return collection

# RESPONSE GENERATION
def create_simple_prompt(query: str, products: list, intent: dict) -> str:
    """
    Creates a structured prompt for OpenAI model based on the user's query, extracted intent, and shortlisted products.
    """

    prompt = f"""The user searched for: "{query}"

Their intent includes:
- Category: {intent.get('category', 'N/A')}
- Brand: {intent.get('brand', 'N/A')}
- Style: {intent.get('style', 'N/A')}
- Colors: {', '.join([attr['value'] for attr in intent.get('attributes', []) if attr['type'] == 'color']) or 'N/A'}

Here are the top {len(products)} products retrieved:
"""

    for i, product in enumerate(products, 1):
        title = product.get('product_title', 'N/A')
        brand = product.get('manufacturer_brand', 'N/A')
        color = product.get('primary_color', 'N/A')
        material = product.get('primary_material', 'N/A')

        features = product.get('product_features', '')
        if '.' in features:
            features_summary = features.split('.')[0] + '.'
        else:
            features_summary = features[:100] + ('...' if len(features) > 100 else '')

        prompt += f"""
Product {i}:
- Title: {title}
- Brand: {brand}
- Color: {color}
- Material: {material}
- Key Feature: {features_summary}
"""

    # Add visual divider before the task begins
    prompt += """
---
🛍️ ShopTalk's Recommendation Below ⬇️
"""

    prompt += f"""

Your Task:
1. Summarize each product in 1–2 lines.
2. Highlight what makes each one unique (e.g., design, feature, brand).
3. If the user cares most about color, brand, or material (as inferred from their query), guide them accordingly.
4. Use friendly, direct tone.
5. End with a suggestion — recommend which product is best if user prefers X, and which if they prefer Y.

Keep it tight, helpful, and structured.
"""

    return prompt.strip()




MAX_DOLLAR_PER_CALL = 0.002  # Estimated ceiling

def generate_openai_response(prompt: str, products: list, model="gpt-3.5-turbo") -> str:
    # Smart token budgeting
    max_tokens = min(350, 100 + len(products) * 50)  # Max cap at 350
    per_product_token_budget = max(30, (max_tokens - 100) // len(products))  # Ensure at least 30 tokens per product

    estimated_cost = (max_tokens * 2 / 1000) * 0.0015  # input + output
    print(f"Estimated OpenAI call cost: ${estimated_cost:.6f} | Max tokens: {max_tokens} | Per product: {per_product_token_budget}")

    if estimated_cost > MAX_DOLLAR_PER_CALL:
        print("Skipping OpenAI call: estimated cost exceeds safe limit.")
        return "Response skipped to stay within cost limits."

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You're an expert shopping assistant. When responding:\n"
                        "- Be clear, concise, and friendly.\n"
                        "- Compare the products retrieved based on user interest (color, material, brand, etc).\n"
                        "- Highlight what makes each product unique.\n"
                        "- Reference what the user is looking for (e.g., 'Since you're looking for red sneakers...').\n"
                        "- Suggest different options based on different preferences.\n"
                        "- End with a thoughtful recommendation."
                    )
                },
                {
                    "role": "user",
                    "content": (
                        prompt + 
                        f"\nYou have ~{per_product_token_budget} tokens per product.\n"
                        "Your task:\n"
                        "1. Summarize each product in 1–2 lines.\n"
                        "2. Highlight unique features (material, design, etc).\n"
                        "3. Suggest which product fits which kind of user based on their needs.\n"
                        "4. End with a clear recommendation based on the query."
                    )
                }
            ],
            # temperature range is 0 to 1 
            # where 0.0 - 0.3 is deterministic; 0.4 - 0.7 is balance; 0.8 - 1.0 is creative/risky
            # recommended is 0.6
            temperature=0.8,

            # top_p range is 0 to 1 
            # where 0.0 - 0.3 is strict/robotic; 0.4 - 0.85 is balanced control; 0.9 - 1.0 is open/creative
            # recommended is 0.85
            top_p=0.9,

            max_tokens=max_tokens
        )

        return response.choices[0].message.content.strip()

    except Exception as e:
        print(f"❌ OpenAI API error: {e}")
        return "I'm having trouble generating a response right now."



# API MODELS
# Request/Response Models
class QueryRequest(BaseModel):
    """
    Request model for product queries
    """
    query: str = Field(..., description="Natural language query to search for products")
class QueryResponse(BaseModel):
    response: str
    products: List[Dict[str, Any]]
    using_bm25: bool
    
# API ENDPOINTS
# Main Query Endpoint
@routerRag.post("/query", response_model=QueryResponse)
async def query_milvus(request: QueryRequest, collection_name: str):
    """
    Query products based on natural language input.
    
    Returns relevant products and a conversational response.
    """
    try:
        if not utility.has_collection(collection_name):
            raise HTTPException(status_code=404, detail=f"Collection '{collection_name}' not found.")
        
        collection = Collection(collection_name)
        collection.load()
        
        # Get BM25 model from handler
        bm25_model, corpus_mapping = get_bm25_model(collection_name, DATA_PATH)
        if bm25_model is not None:
            print("Using BM25 model")
        else:
            print("No BM25 model available for this collection")
        
        # Format query with structure and get intent in one step
        structured_query, intent = process_query(request.query)
        print(f"Query intent: {intent}")
        
        # Generate embedding for the vector search
        query_embedding = generate_embeddings([structured_query])[0].tolist()
        
        # Perform vector search with Milvus
        search_params = {"metric_type": "IP", "params": {"nprobe": 10}}
        
        # Define output fields (using both new and old field names for compatibility)
        output_fields = ALL_FIELDS.copy() 
        
        # Prepare expression for filtering (if category is specified)
        expr = None
        if intent["category"]:
            # Try exact match first
            expr = f'product_category == "{intent["category"]}"'
            try:
                has_results = collection.query(expr=expr, limit=1)
                if not has_results:
                    expr = None
            except Exception as e:
                print(f"Error with expression query: {e}")
                expr = None
        
        # First-stage retrieval from Milvus
        try:
            results = collection.search(
                [query_embedding],
                "embedding",
                search_params,
                limit=10,  # Increase to 10 to get more candidates for re-ranking
                expr=expr,
                output_fields=output_fields
            )
        except Exception as e:
            print(f"Search error with expression: {e}")
            # Try again without expression if there was an error
            results = collection.search(
                [query_embedding],
                "embedding",
                search_params,
                limit=10,  # Increase to 10 for re-ranking
                output_fields=output_fields
            )

        # NEW CODE: Perform BM25 search if model is available
        bm25_results_dict = {}
        if bm25_model and corpus_mapping:
            try:
                # Use the BM25 handler for searching
                if bm25_model and corpus_mapping:
                    bm25_results_dict = search_bm25(
                        collection_name=collection_name,
                        query=request.query,
                        bm25_model=bm25_model,
                        corpus_mapping=corpus_mapping,
                        top_k=10
                    )
                    print(f"BM25 found {len(bm25_results_dict)} matching products") 
            except Exception as e:
                print(f"Error in BM25 search: {e}")
                # Continue without BM25 results

        # Check if we have vector results
        if not results or len(results) == 0 or len(results[0]) == 0:
            return {"response": "I couldn't find any products matching your search. Could you try with different keywords or browse our popular categories like phone cases, shoes, or books?", "products": []}

        # Extract entities for re-ranking
        candidate_docs = []
        for hit in results[0]:
            try:
                # Convert hit.entity to dict
                entity_dict = {}
                for field in output_fields:
                    try:
                        # Try dictionary-style access first (most common)
                        value = hit.entity[field]
                        if value not in ["nan", "None", "", None]:
                            entity_dict[field] = value
                    except (KeyError, TypeError):
                        try:
                            # Try attribute access as fallback
                            if hasattr(hit.entity, field):
                                value = getattr(hit.entity, field)
                                if value not in ["nan", "None", "", None]:
                                    entity_dict[field] = value
                        except:
                            # Field not accessible, skip it
                            pass
                
                # Store original vector score
                entity_dict["_vector_score"] = hit.score
                candidate_docs.append(entity_dict)
            except Exception as e:
                print(f"Error processing hit: {e}")        

        # Use the extracted reranking function for hybrid scoring and filtering
        filtered_docs = rank_documents(
            query=request.query,
            candidate_docs=candidate_docs,
            bm25_results_dict=bm25_results_dict,
            top_k=3
        )

        # If no docs were found after re-ranking, return empty response
        if not filtered_docs:
            return {
                "response": "I couldn't find any products matching your search. Could you try with different keywords?", 
                "products": []
            }
        
        # Extract relevant product information
        retrieved_docs = []
        products = []

        for doc in filtered_docs:
            # Remove internal scoring field
            if "_vector_score" in doc:
                del doc["_vector_score"]
            
            # Keep doc as is for retrieved_docs
            retrieved_docs.append(doc)
            
            # Create a product representation with fallbacks for missing fields
            product = {
                "product_id": doc.get("product_id", "unknown_id"),
                "product_title": doc.get("product_title", "Unknown Product"),
                "product_category": doc.get("product_category", "Unknown Category"),
                "manufacturer_brand": doc.get("manufacturer_brand", ""),
                "primary_color": doc.get("primary_color", ""),
                # "product_features": str(doc.get("product_features", ""))[:200] + "...",
                "product_features": str(doc.get("product_features", ""))[:400] + ("..." if len(doc.get("product_features", "")) > 400 else ""),
                "primary_image_id": doc.get("primary_image_id", ""),
                "image_storage_path": doc.get("image_storage_path", "")
            }
            products.append(product)
        
        # Create prompt using the simplified function
        prompt = create_simple_prompt(request.query, products, intent)

        # Log the prompt for debugging
        print(f"LLM prompt preview: {prompt[:200]}...")

        # # Generate recommendation with appropriate parameters
        # llm_response = llm(
        #     prompt,
        #     max_length=512,          # Significantly increased from 150
        #     min_length=100,          # Ensure we get a reasonable response
        #     do_sample=True,          # Enable sampling for more natural text
        #     temperature=0.7,         # Good balance of creativity and focus
        #     top_p=0.9,               # Nucleus sampling to prevent nonsense
        #     top_k=50,                # Consider top 50 tokens at each step
        #     repetition_penalty=1.1,  # Discourage repetitive text
        #     num_return_sequences=1
        # )
        
        response = generate_openai_response(prompt, products)
 
        # Return both the AI response and retrieved documents
        return {
            # "response": llm_response[0]['generated_text'].strip(),
            "response": response,
            "products": products,
            "using_bm25": bm25_model is not None and len(bm25_results_dict) > 0
        }
        
    except Exception as e:
        print(f"Error in query: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error querying Milvus: {e}")

# Document Management Endpoints
@routerRag.get("/documents")
async def list_documents():
    """List all files in the data directory."""
    try:
        files = [f for f in os.listdir(DATA_PATH) if os.path.isfile(os.path.join(DATA_PATH, f))]
        return {"documents": files}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@routerRag.post("/upload_file")
async def upload_file(file: UploadFile = File(...)):
    """Upload a file to the server."""
    try:
        file_location = os.path.join(DATA_PATH, file.filename)
        with open(file_location, "wb") as f:
            f.write(await file.read())
        return {"message": f"File '{file.filename}' uploaded successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error uploading file: {e}")

@routerRag.post("/process_documents")
async def process_documents_endpoint(collection_name: str, file_name: str = None):
    """Process documents and insert them into Milvus collection."""
    try:
        process_documents(collection_name, file_name)
        
        # Clear cached BM25 model after processing new documents
        clear_bm25_cache(collection_name)
            
        return {"message": f"Documents processed and inserted into Milvus collection '{collection_name}'."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing documents: {e}")

# System management Endpoints
@routerRag.delete("/delete_milvus_index")
async def delete_milvus_index(collection_name: str):
    """Delete a Milvus collection."""
    try:
        if utility.has_collection(collection_name):
            utility.drop_collection(collection_name)
            return {"message": f"Milvus collection '{collection_name}' deleted successfully."}
        else:
            return {"message": f"Collection '{collection_name}' does not exist."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting Milvus index: {e}")

@routerRag.delete("/delete_file")
async def delete_file(filename: str):
    """Delete a file from the data directory."""
    try:
        file_path = os.path.join(DATA_PATH, filename)
        if os.path.exists(file_path):
            os.remove(file_path)
            return {"message": f"File '{filename}' deleted successfully."}
        else:
            raise HTTPException(status_code=404, detail=f"File '{filename}' not found.")
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting file: {e}")

@routerRag.get("/check_collection")
async def check_collection(collection_name: str):
    """Check if a collection exists and get its stats."""
    try:
        if utility.has_collection(collection_name):
            collection = Collection(collection_name)
            collection.load()
            return {
                "exists": True,
                "row_count": collection.num_entities,
                "fields": [field.name for field in collection.schema.fields]
            }
        else:
            return {"exists": False}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
  