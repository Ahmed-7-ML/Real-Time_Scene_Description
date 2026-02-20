import torch
from transformers import (
    AutoProcessor, AutoModelForCausalLM,
    BlipProcessor, BlipForConditionalGeneration,
    ViTImageProcessor, VisionEncoderDecoderModel,
    AutoTokenizer
)

# --- CONFIGURATION: Safety Tokens & Control ---
SAFETY_TOKENS = [
    "hole", "holes", "open hole",
    "puddle", "puddles",
    "mud", "muddy",
    "flooded", "flood",
    "construction", "construction site",
    "debris", "rubble",
    "broken", "damaged", "crack", "uneven",
    "obstacle", "blocked", "barrier",
    "fire", "smoke",
    "accident", "crash",
    "train", "railway", "tracks",
    "scaffolding", "tools",
    "icy", "snow", "slippery",
    "trash", "garbage", "clutter"
]

hazards_str = ', '.join(SAFETY_TOKENS)
CAPTION_PROMPT = f"Caption: "

def load_git(device="cpu"):
    print("Loading GIT model...")
    processor = AutoProcessor.from_pretrained("microsoft/git-base")
    model = AutoModelForCausalLM.from_pretrained("microsoft/git-base").to(device)
    return processor, model

def load_blip(device="cpu"):
    print("Loading BLIP model...")
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)
    return processor, model

def load_gpt2(device="cpu"):
    print("Loading ViT-GPT2 model...")
    processor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning").to(device)
    return processor, model, tokenizer

def generate_caption(model_name, image, processors, models, device="cpu"):
    if model_name == "GIT":
        processor, model = processors["GIT"], models["GIT"]
        inputs = processor(images=image, text=CAPTION_PROMPT, return_tensors="pt").to(device)
        ids = model.generate(pixel_values=inputs.pixel_values, input_ids=inputs.input_ids, max_new_tokens=50)
        return processor.batch_decode(ids, skip_special_tokens=True)[0]
    
    elif model_name == "BLIP":
        processor, model = processors["BLIP"], models["BLIP"]
        inputs = processor(images=image, text=CAPTION_PROMPT, return_tensors="pt").to(device)
        ids = model.generate(**inputs, max_new_tokens=50)
        return processor.decode(ids[0], skip_special_tokens=True)
    
    elif model_name == "ViT-GPT2":
        processor, model, tokenizer = processors["GPT2"], models["GPT2"], processors["GPT2_tokenizer"]
        inputs = processor(images=image, return_tensors="pt").to(device)
        prompt_ids = tokenizer(CAPTION_PROMPT, return_tensors="pt").input_ids.to(device)
        ids = model.generate(pixel_values=inputs.pixel_values, decoder_input_ids=prompt_ids, max_new_tokens=50)
        return tokenizer.batch_decode(ids, skip_special_tokens=True)[0]
    
    return ""
