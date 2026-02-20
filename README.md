---
title: Scene Safety Captioner
emoji: 🛡️
colorFrom: red
colorTo: green
sdk: gradio
sdk_version: 4.19.2
app_file: app.py
pinned: false
---

# Scene Safety Captioner 🛡️

This application uses Image Captioning models (GIT, BLIP, ViT-GPT2) combined with a **Smart Safety Classifier (Regex-based)** to identify potential hazards for visually impaired individuals.

## Key Features
- **Multi-Model Support**: Compare results from GIT, BLIP, and ViT-GPT2.
- **Hazard Token Control**: Models are prompted with specific hazard vocabulary to increase sensitivity to dangers.
- **Smart Safety Logic**: Uses Version 2 of our Regex Classifier, which understands:
    - **Negation**: "No potholes" is marked as SAFE.
    - **Proximity**: "Construction far away" is marked as SAFE.
    - **Context**: "Toy fire" or "Picture of an accident" is marked as SAFE.
    - **Segments**: Handles complex sentences with "but" or "however".

## Models Used
- `microsoft/git-base`
- `Salesforce/blip-image-captioning-base`
- `nlpconnect/vit-gpt2-image-captioning`
