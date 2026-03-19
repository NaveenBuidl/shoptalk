from fastapi import FastAPI, WebSocket, WebSocketDisconnect
#from auth import get_current_user  # Import authentication function
from pymilvus import connections
import os
from tqdm import tqdm
import glob
import pandas as pd
import json
import numpy as np
import re
import os
import glob
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import shutil
from sqlalchemy.orm import Session
from fastapi.security import OAuth2PasswordBearer
#from database import get_db, Base, engine
#from models import User
#from schemas import UserCreate, UserLogin, Token
#from auth import get_current_user, authenticate_user, create_access_token, get_password_hash
from datetime import timedelta
#from auth import router as auth_router
# from app.rag_02042025 import routerRag as rag_router
# from app.rag_03042025 import routerRag as rag_router
from rag import routerRag as rag_router
import logging
from typing import List, Dict
# Add this to the beginning of your app.py file
# Add to the top of your app.py
import os
import nltk
nltk_data_dir = os.path.join(os.getcwd(), "nltk_data")
os.makedirs(nltk_data_dir, exist_ok=True)
nltk.data.path.append(nltk_data_dir)
nltk.download('punkt', download_dir=nltk_data_dir)
nltk.download('punkt_tab', download_dir=nltk_data_dir, quiet=True)
nltk.download('stopwords', download_dir=nltk_data_dir)


# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
# Get the external Milvus details from environment variables
#MILVUS_HOST = os.getenv("MILVUS_HOST","milvus-standalone")
MILVUS_HOST = os.getenv("MILVUS_HOST","localhost")
MILVUS_PORT = os.getenv("MILVUS_PORT", "19530")

# Directory for storing files
DATA_PATH = "/app/data"
os.makedirs(DATA_PATH, exist_ok=True)  # Ensure the directory exists
# Initialize FastAPI app
app = FastAPI()

active_connections = []

# Allow all origins (for testing only, not recommended for production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins # Adjust for production ( allow only from crtain endpoints)
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allows all headers
)


# @app.get("/")
# def serve_react():
#    return FileResponse("./frontend/public/index.html")
# Create database tables
#Base.metadata.create_all(bind=engine)

#oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")


# @app.get("/protected/")
# def protected_route(current_user: User = Depends(get_current_user)):
#     return {"message": f"Hello, {current_user.username}! This is a protected route."}


# Connect to Milvus
try:
    connections.connect("default", host=MILVUS_HOST, port=MILVUS_PORT)
    print("Successfully connected to Milvus")
except Exception as e:
    print(f"Failed to connect to Milvus: {e}") 

 


#app.include_router(auth_router, prefix="/auth")
app.include_router(rag_router, prefix="/rag")


# # Test embedding to get the dimension
# test_embedding = emb_text("This is a test")
# embedding_dim = len(test_embedding)
# print(f"Embedding Dimension: {embedding_dim}")
# print(f"Sample Embedding: {test_embedding[:10]}")



# @app.get("/")
# async def root():
#     return {"message": "Hello World"}

