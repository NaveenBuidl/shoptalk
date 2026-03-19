import streamlit as st
import requests
import os
from PIL import Image
import boto3


# AWS and Backend Configuration
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
S3_BUCKET = os.getenv("S3_BUCKET")
S3_REGION = os.getenv("S3_REGION", "us-east-1")
BACKEND_URL = os.getenv("BACKEND_URL")
s3_client = boto3.client(
    's3',
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    region_name=S3_REGION
)


# Configuration

LOCAL_IMAGE_PATH = os.getenv("LOCAL_IMAGE_PATH", "./images/small_selected")
USE_S3 = True  # Set to True if using S3
# USE_S3 = True
#S3_BUCKET = "shoptalk-product-images"

def fetch_image(image_path, max_size=(100, 100)):
    """Fetch image from local storage or S3."""
    try:
        if USE_S3:
            # Ensure path is properly formatted
            s3_path = image_path.lstrip("/")
            s3_path = f"selected_images/{s3_path}"
            
            #st.write(f"Trying to load from S3: Bucket={S3_BUCKET}, Key={s3_path}")
            
            obj = s3_client.get_object(Bucket=S3_BUCKET, Key=s3_path)
            return Image.open(obj["Body"])
        else:
            return fetch_local_image(image_path)
    except Exception as e:
        st.error(f"Error loading image: {e}")
        return None
    
def fetch_local_image(image_path):
    # Normalize and isolate the last two components (e.g., '74/74613e89.jpg')
    image_path = image_path.replace("\\", "/")
    parts = image_path.strip("/").split("/")
    if len(parts) >= 2:
        rel_path = os.path.join(parts[-2], parts[-1])
    else:
        rel_path = parts[-1]  # fallback
    
    full_path = os.path.join(LOCAL_IMAGE_PATH, rel_path)
    
    # st.write(f"🛠️ Looking for image at: {full_path}")  # Debug
    try:
        img = Image.open(full_path)
        return img
    except FileNotFoundError:
        st.error(f"Image not found: {full_path}")
        return None
    except Exception as e:
        st.error(f"Error loading image: {e}")
        return None


def query_backend(query):
    """Send query to FastAPI backend and return response."""
    response = requests.post(BACKEND_URL, json={"query": query})
    return response.json()

# Streamlit UI
st.title("📦 ShopTalk Product Search")

query = st.text_input("🔍 Enter your query:")

if st.button("Search"):
    if query:
        response = query_backend(query)
        products = response.get("products", [])

        if "response" in response:
            st.markdown("---")
            st.subheader("🤖 ShopTalk Chat")
            
            llm_response = response["response"]
            # Separate the final recommendation
            if "\n" in llm_response:
                main_part, final_part = llm_response.rsplit("\n", 1)
            else:
                main_part, final_part = "", llm_response

            st.write(main_part)

            # Quote-style block with light yellow and plain italics
            st.markdown(f"""
            <div style='
                background-color: #fffbea;
                padding: 1em;
                margin-top: 1em;
                border-left: 5px solid #f2d874;
                border-radius: 6px;
                font-style: italic;
                font-size: 16px;
                color: #444;
            '>
            {final_part}
            </div>
            """, unsafe_allow_html=True)
 
        # Check if retrieved documents are available
        # if "retrieved_documents" in response:
        if "products":

            for item in products:
                st.markdown("---")  # Separator for readability
                col1, col2 = st.columns([1, 2])  # Image (1/3), Text (2/3)

                with col1:
                    image = fetch_image(item["image_storage_path"])
                    if image:
                        st.image(image, use_container_width=True)
                    else:
                        st.write("Image not available")

                with col2:
                    st.markdown(f"<p style='font-size:18px;'><b>🧢 Product ID:</b> {item['product_id']}</p>", unsafe_allow_html=True)
                    st.markdown(f"<p style='font-size:18px;'><b>🧢 Product Title:</b> {item['product_title']}</p>", unsafe_allow_html=True)
                    st.markdown(f"<p style='font-size:18px;'><b>🏷️ Product Category:</b> {item['product_category']}</p>", unsafe_allow_html=True)
                    #st.markdown(f"<p style='font-size:18px;'><b>🖼️ Product Image ID:</b> {item['primary_image_id']}</p>", unsafe_allow_html=True)
                    st.markdown(f"<p style='font-size:18px;'><b>📄 Product Features:</b> {item['product_features']}</p>", unsafe_allow_html=True)
        
        else:
            # st.warning("No results found.")
            # Clean fallback if no results
            st.markdown("<h4 style='text-align: center;'>😕 I couldn't find any matching products.</h4>", unsafe_allow_html=True)
            st.markdown("<p style='text-align: center;'>💡 Popular searches: <code>shoes</code>, <code>chair</code>, <code>sofa</code>, <code>phone case</code>, <code>grocery</code>.</p>", unsafe_allow_html=True)

    else:
        st.error("Please enter a query.")
        
        