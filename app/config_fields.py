# config_fields.py
"""
Configuration file for field ordering and properties in the ShopTalk application.
This centralizes field definitions to make experimentation easier.
"""
import os
from config_embedding import EMBEDDING_DIMENSION

# Define field structure with ordering and max_length properties
FIELDS_CONFIG = {
    # ID fields (will be processed first)
    'product_id': {'order': 1, 'max_length': 100, 'is_id': True, 'is_primary': True},
    'primary_image_id': {'order': 2, 'max_length': 100, 'is_id': True},
    'image_storage_path': {'order': 3, 'max_length': 100, 'is_id': True},
    
    # Core embedding fields in priority order
    'product_title': {'order': 4, 'max_length': 500, 'is_id': False},
    'product_category': {'order': 5, 'max_length': 100, 'is_id': False},
    'product_features': {'order': 6, 'max_length': 4000, 'is_id': False},
    'long_description': {'order': 7, 'max_length': 4000, 'is_id': False},
    'visual_description': {'order': 8, 'max_length': 500, 'is_id': False},
    'manufacturer_brand': {'order': 9, 'max_length': 100, 'is_id': False},
    'product_model': {'order': 10, 'max_length': 200, 'is_id': False},
    'category_path': {'order': 11, 'max_length': 500, 'is_id': False},
    'primary_color': {'order': 12, 'max_length': 100, 'is_id': False},
    'design_style': {'order': 13, 'max_length': 100, 'is_id': False},
    'primary_material': {'order': 14, 'max_length': 100, 'is_id': False},
    'textile_material': {'order': 15, 'max_length': 100, 'is_id': False},
    'design_pattern': {'order': 16, 'max_length': 100, 'is_id': False},
    'physical_shape': {'order': 17, 'max_length': 100, 'is_id': False},
    'surface_finish': {'order': 18, 'max_length': 100, 'is_id': False},
    'search_terms': {'order': 19, 'max_length': 4000, 'is_id': False},
}

# Generate lists based on the config for easier access
ID_COLUMNS = [field for field, props in sorted(
    [(f, p) for f, p in FIELDS_CONFIG.items() if p['is_id']], 
    key=lambda x: x[1]['order']
)]

EMBEDDING_PRIORITY_COLUMNS = [field for field, props in sorted(
    [(f, p) for f, p in FIELDS_CONFIG.items() if not p['is_id']], 
    key=lambda x: x[1]['order']
)]

# Get all fields in schema order
ALL_FIELDS = [field for field, props in sorted(
    FIELDS_CONFIG.items(), 
    key=lambda x: x[1]['order']
)]

# Field mapping for backward compatibility with original ABO dataset names
ORIGINAL_TO_NEW_MAPPING = {
    "item_id": "product_id",
    "image_id": "primary_image_id",
    "image_path": "image_storage_path",
    # Order of the fields
    "item_name": "product_title",
    "product_type": "product_category",
    "bullet_point": "product_features",
    "product_description": "long_description",
    "image_caption": "visual_description",
    "brand": "manufacturer_brand",
    "model_name": "product_model",
    "node_name": "category_path",
    "color_standardized": "primary_color",
    "style": "design_style",
    "material": "primary_material",
    "fabric_type": "textile_material",
    "pattern": "design_pattern",
    "item_shape": "physical_shape",
    "finish_type": "surface_finish",
    "item_keywords": "search_terms",
}

# Reverse mapping for backward compatibility
NEW_TO_ORIGINAL_MAPPING = {v: k for k, v in ORIGINAL_TO_NEW_MAPPING.items()}

# Function to modify field order during experimentation
def reorder_fields(new_order_list):
    """
    Temporarily reorder fields for experimentation.
    
    Args:
        new_order_list: List of field names in desired order
    
    Returns:
        Original order to restore later
    """
    global EMBEDDING_PRIORITY_COLUMNS
    old_order = EMBEDDING_PRIORITY_COLUMNS.copy()
    
    # Update only non-ID fields
    non_id_fields = [field for field in new_order_list 
                    if field in FIELDS_CONFIG and not FIELDS_CONFIG[field]['is_id']]
    
    EMBEDDING_PRIORITY_COLUMNS = non_id_fields
    return old_order
