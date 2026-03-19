import streamlit as st
import boto3
from PIL import Image
import io
import os
def test_specific_s3_image():
    st.title("S3 Image Access Test")
    AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
    AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
    S3_BUCKET = os.getenv("S3_BUCKET")
    S3_REGION = os.getenv("S3_REGION", "us-east-1")
    BACKEND_URL = os.getenv("BACKEND_URL")
    # S3 credentials
    aws_access_key = st.text_input("AWS Access Key", AWS_ACCESS_KEY_ID)
    aws_secret_key = st.text_input("AWS Secret Key", AWS_SECRET_ACCESS_KEY, type="password")
    region = st.text_input("AWS Region", S3_REGION)
    
    # S3 bucket and object info
    bucket_name = st.text_input("S3 Bucket Name", S3_BUCKET)
    object_key = st.text_input("Object Key (Path)", "small_selected/eb/eb817105.jpg")
    
    # Create S3 client on demand
    if st.button("Test S3 Access"):
        try:
            # Create S3 client
            s3_client = boto3.client(
                's3',
                aws_access_key_id=aws_access_key,
                aws_secret_access_key=aws_secret_key,
                region_name=region
            )
            
            # Try listing objects first
            st.subheader("Testing bucket access...")
            try:
                response = s3_client.list_objects_v2(
                    Bucket=bucket_name,
                    MaxKeys=5  # Just get a few objects
                )
                
                if 'Contents' in response:
                    st.success(f"✅ Successfully listed objects in bucket '{bucket_name}'")
                    sample_objects = [item['Key'] for item in response['Contents']]
                    st.write("Sample objects in bucket:")
                    for obj in sample_objects:
                        st.write(f"- {obj}")
                else:
                    st.warning(f"⚠️ Bucket '{bucket_name}' exists but appears to be empty")
            except Exception as list_error:
                st.error(f"❌ Error listing objects: {type(list_error).__name__}: {str(list_error)}")
            
            # Try getting the specific object
            st.subheader("Testing specific object access...")
            try:
                st.write(f"Trying to access: {bucket_name}/{object_key}")
                obj = s3_client.get_object(Bucket=bucket_name, Key=object_key)
                
                # Check content type
                content_type = obj.get('ContentType', 'unknown')
                st.write(f"Content type: {content_type}")
                
                # If it's an image, display it
                if content_type.startswith('image/'):
                    image_data = obj['Body'].read()
                    image = Image.open(io.BytesIO(image_data))
                    st.success(f"✅ Successfully loaded image: {object_key}")
                    st.image(image, caption=f"Image from {bucket_name}/{object_key}")
                    
                    # Show image details
                    st.write(f"Image size: {image.size}")
                    st.write(f"Image format: {image.format}")
                else:
                    st.warning(f"⚠️ Object was found but is not an image (content type: {content_type})")
                    
            except Exception as obj_error:
                st.error(f"❌ Error getting object: {type(obj_error).__name__}: {str(obj_error)}")
                
            # Suggest a publicly accessible URL (if the bucket allows it)
            public_url = f"https://{bucket_name}.s3.{region}.amazonaws.com/{object_key}"
            st.subheader("Public URL (if bucket allows public access):")
            st.write(public_url)
            
        except Exception as e:
            st.error(f"❌ Error initializing S3 client: {type(e).__name__}: {str(e)}")

if __name__ == "__main__":
    test_specific_s3_image()