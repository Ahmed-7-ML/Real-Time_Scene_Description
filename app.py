import gradio as gr
import torch
from PIL import Image
import os
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from classifier import SmartSafetyClassifierV2
from model_utils import load_git, load_blip, load_gpt2, generate_caption

# --- Initialization ---
device = "cuda" if torch.cuda.is_available() else "cpu"
classifier = SmartSafetyClassifierV2()

# Lazy loading containers
models = {}
processors = {}

def get_models(model_name):
    if model_name not in models:
        if model_name == "GIT":
            p, m = load_git(device)
            processors["GIT"], models["GIT"] = p, m
        elif model_name == "BLIP":
            p, m = load_blip(device)
            processors["BLIP"], models["BLIP"] = p, m
        elif model_name == "ViT-GPT2":
            p, m, t = load_gpt2(device)
            processors["GPT2"], models["GPT2"] = p, m
            processors["GPT2_tokenizer"] = t
    return processors, models

def predict(image, model_choice):
    if image is None:
        return "Please upload an image.", "Unknown"
    
    # Load model if needed
    get_models(model_choice)
    
    # Generate caption
    raw_caption = generate_caption(model_choice, image, processors, models, device)
    
    # Since vocabulary prompt is removed, we just need to handle potential 
    # model echoes of the instruction "a photo of" or similar.
    # GIT and GPT2 sometimes repeat the input prompt.
    
    caption = raw_caption.strip()
    
    # Heuristic: if the caption starts with our prompt, remove it
    from model_utils import CAPTION_PROMPT
    p = CAPTION_PROMPT.strip().lower()
    if caption.lower().startswith(p):
        caption = caption[len(p):].strip()
        # Remove leading punctuation/noise if any
        caption = re.sub(r'^[^a-zA-Z0-9]+', '', caption).strip()

    # Classify safety
    safety_status = classifier.classify(caption)
    
    color = "🟢 SAFE" if safety_status == "SAFE" else "🔴 DANGEROUS"
    
    return caption, color

# --- UI Design ---
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🛡️ Scene Safety & Image Captioning")
    gr.Markdown("Identify hazards in images using AI. Select a model and upload an image to get a safety-aware caption.")
    
    with gr.Row():
        with gr.Column():
            input_img = gr.Image(type="pil", label="Upload Scene")
            model_drop = gr.Dropdown(
                choices=["GIT", "BLIP", "ViT-GPT2"], 
                value="GIT", 
                label="Select Model"
            )
            submit_btn = gr.Button("Analyze Scene", variant="primary")
        
        with gr.Column():
            output_caption = gr.Textbox(label="Generated Caption")
            output_safety = gr.Label(label="Safety Assessment")
            
    gr.Examples(
        examples=[], # You can add local paths to images if available
        inputs=input_img
    )
    
    submit_btn.click(
        fn=predict,
        inputs=[input_img, model_drop],
        outputs=[output_caption, output_safety]
    )

if __name__ == "__main__":
    demo.launch()
