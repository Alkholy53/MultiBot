#import clip
import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import streamlit as st


# Load the CLIP model for image captioning
device = "cpu"
# Load the BLIP model for dynamic image captioning
blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)

# Function to generate image captions
def generate_caption(image_path):
    try:
        image = Image.open(image_path)
        inputs = blip_processor(images=image, return_tensors="pt").to(device)
        out = blip_model.generate(**inputs)
        caption = blip_processor.decode(out[0], skip_special_tokens=True)
        return caption
    except Exception as e:
        st.write(f"Error in generating caption: {e}")
        return "Unable to generate caption for the provided image."
