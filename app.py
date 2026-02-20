import gradio as gr
import torch
import cv2
from PIL import Image
import numpy as np
import os
import sys
import tempfile

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

def predict_caption(image, model_choice):
    """Core translation from Image -> Caption & Color"""
    if image is None:
        return "Please provide an image.", "Unknown"
    
    # Load model if needed
    get_models(model_choice)
    
    # Generate caption
    raw_caption = generate_caption(model_choice, image, processors, models, device)
    
    # Post-processing the caption
    import re
    caption = raw_caption.strip()
    noise_patterns = [
        r'^generate\s+a\s+description\s+for\s+this\s+image\.?',
        r'^caption\s*:\s*',
        r'^a\s+photo\s+of\s+',
        r'^an\s+image\s+of\s+'
    ]
    for pattern in noise_patterns:
        caption = re.sub(pattern, '', caption, flags=re.IGNORECASE).strip()
    if "caption:" in caption.lower():
        parts = re.split(r'caption\s*:\s*', caption, flags=re.IGNORECASE)
        caption = parts[-1].strip()
    caption = re.sub(r'^[^a-zA-Z0-9]+', '', caption).strip()

    # Classify safety
    safety_status = classifier.classify(caption)
    color = "🟢 SAFE" if safety_status == "SAFE" else "🔴 DANGEROUS"
    
    return caption, color

def process_video(video_path, model_choice, progress=gr.Progress()):
    if not video_path:
        return None, "Please upload a video."
    
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or fps != fps:
        fps = 30
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Process 1 frame per second to maintain performance
    frame_skip = int(fps)
    if frame_skip == 0:
        frame_skip = 1
        
    out_path = tempfile.mktemp(suffix=".mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
    
    current_caption = "Analyzing..."
    current_color = "🟢 SAFE"
    
    frames_processed = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        if frames_processed % frame_skip == 0:
            # Convert frame BGR to PIL Image
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(frame_rgb)
            # Run inference
            caption, color = predict_caption(pil_img, model_choice)
            current_caption = caption
            current_color = color
            
        # Draw on frame
        color_bgr = (0, 255, 0) if "SAFE" in current_color else (0, 0, 255)
        text_display = f"{current_color.replace('🟢 ', '').replace('🔴 ', '')}: {current_caption}"
        
        # Adding text background
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        thickness = 2
        (text_width, text_height), baseline = cv2.getTextSize(text_display, font, font_scale, thickness)
        
        cv2.rectangle(frame, (10, 10), (10 + text_width, 10 + text_height + baseline + 10), (0, 0, 0), -1)
        cv2.putText(frame, text_display, (10, 10 + text_height + 5), font, font_scale, color_bgr, thickness)
        
        out.write(frame)
        frames_processed += 1
        
        if frames_processed % frame_skip == 0:
            progress(frames_processed / max(1, frame_count), desc="Processing Video Frame-by-Frame...")
            
    cap.release()
    out.release()
    
    return out_path, "Processing Complete."

# --- UI Design ---
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🛡️ Real-Time Scene Description & Safety Assessment")
    gr.Markdown("Identify hazards in various media formats using deep learning. Choose an input format below.")
    
    # Top level model selection
    with gr.Row():
        global_model_drop = gr.Dropdown(
            choices=["GIT", "BLIP", "ViT-GPT2"], 
            value="GIT", 
            label="🌍 Select Global Image Captioning Model",
            info="This model applies to all tabs."
        )

    with gr.Tabs():
        # TAB 1: Image
        with gr.TabItem("🖼️ Image Analysis"):
            with gr.Row():
                with gr.Column():
                    img_input = gr.Image(type="pil", label="Upload Scene Image")
                    img_btn = gr.Button("Analyze Scene", variant="primary")
                with gr.Column():
                    img_output_caption = gr.Textbox(label="Generated Caption")
                    img_output_safety = gr.Label(label="Safety Assessment")
                    
            img_btn.click(
                fn=predict_caption,
                inputs=[img_input, global_model_drop],
                outputs=[img_output_caption, img_output_safety]
            )
            
        # TAB 2: Video
        with gr.TabItem("🎥 Video Processing"):
            with gr.Row():
                with gr.Column():
                    vid_input = gr.Video(label="Upload Video")
                    vid_btn = gr.Button("Analyze Video", variant="primary")
                with gr.Column():
                    vid_output = gr.Video(label="Annotated Output Video", interactive=False)
                    vid_status = gr.Textbox(label="Processing Status")
                    
            vid_btn.click(
                fn=process_video,
                inputs=[vid_input, global_model_drop],
                outputs=[vid_output, vid_status]
            )

        # TAB 3: Live Camera
        with gr.TabItem("🔴 Live Camera"):
            gr.Markdown("**Real-Time Snapshot Mode**: Since continuous inference requires substantial compute, capture a real-time frame from your webcam to have it analyzed locally.")
            with gr.Row():
                with gr.Column():
                    cam_input = gr.Image(sources=["webcam"], type="pil", label="Webcam Stream (Snap an image)")
                    cam_btn = gr.Button("Analyze Snapshot", variant="primary")
                with gr.Column():
                    cam_output_caption = gr.Textbox(label="Real-time Caption")
                    cam_output_safety = gr.Label(label="Safety Assessment")
            
            cam_btn.click(
                fn=predict_caption,
                inputs=[cam_input, global_model_drop],
                outputs=[cam_output_caption, cam_output_safety]
            )

if __name__ == "__main__":
    demo.launch()
