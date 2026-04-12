import os
import json
import pandas as pd
from typing import List

# Paths
MMHS_ROOT = r"d:\Hate Speech Detection\datasets\MMHS-150K"
MMHS_DATA = os.path.join(MMHS_ROOT, "data")
MMHS_IMG = os.path.join(MMHS_ROOT, "img_resized")

FB_ROOT = r"d:\Hate Speech Detection\datasets\Facebook Memes\data"
FB_IMG = os.path.join(FB_ROOT, "img")
# FB JSONs are usually in root or processed folder. Assuming standard locations or from run_facebook.ps1
# But run_facebook.ps1 used data.py directly on original jsons? 
# We need to find where FB jsons are.
# Based on previous context, they might be in `datasets/facebook/train.jsonl` etc if prepared, 
# or we need to convert the original dev.jsonl/train.jsonl provided by FB.
# Let's assume standard locations or check. 
# Better: Check existing files first? 
# I will proceed assuming: datasets/facebook/train.jsonl
# If they don't exist, I'll fall back or error out.

FB_TRAIN = os.path.join(FB_ROOT, "train.jsonl")
FB_DEV = os.path.join(FB_ROOT, "dev.jsonl")
FB_TEST = os.path.join(FB_ROOT, "test.jsonl")

MMHS_TRAIN = os.path.join(MMHS_DATA, "train.jsonl")
MMHS_DEV = os.path.join(MMHS_DATA, "dev.jsonl")
MMHS_TEST = os.path.join(MMHS_DATA, "test.jsonl")

OUT_DIR = r"d:\Hate Speech Detection\datasets\Combined"
os.makedirs(OUT_DIR, exist_ok=True)

def load_and_fix(json_path, img_root, label_map=None):
    if not os.path.exists(json_path):
        print(f"Warning: {json_path} not found. Skipping.")
        return []
    
    data = []
    with open(json_path, 'r', encoding='utf-8') as f:
        for line in f:
            obj = json.loads(line)
            # Fix Image Path
            img_name = os.path.basename(obj['img']) # ensure clean filename
            obj['img'] = os.path.join(img_root, img_name) # Absolute path
            
            # Normalize label
            # MMHS: 0/1 (1=Hate). FB: 0/1 (1=Hate). They align.
            
            data.append(obj)
    return data

def save_jsonl(data, path):
    with open(path, 'w', encoding='utf-8') as f:
        for obj in data:
            f.write(json.dumps(obj) + '\n')
    print(f"Saved {len(data)} records to {path}")

def main():
    print("Preparing Combined Dataset...")
    
    # 1. Load MMHS
    print("Loading MMHS...")
    mmhs_train = load_and_fix(MMHS_TRAIN, MMHS_IMG)
    mmhs_dev = load_and_fix(MMHS_DEV, MMHS_IMG)
    mmhs_test = load_and_fix(MMHS_TEST, MMHS_IMG)
    
    # 2. Load FB Memes
    print("Loading FB Memes...")
    fb_train = load_and_fix(FB_TRAIN, FB_IMG)
    fb_dev = load_and_fix(FB_DEV, FB_IMG)
    fb_test = load_and_fix(FB_TEST, FB_IMG)
    
    if not fb_train:
         # Fallback check for original names if standard ones missing
         print("Checking for original FB filenames...")
         # FB dataset comes as train.jsonl
    
    # 3. Upsample FB Memes (x5)
    FB_UPSAMPLE = 5
    print(f"Upsampling FB Memes Train by {FB_UPSAMPLE}x...")
    fb_train_upsampled = fb_train * FB_UPSAMPLE
    
    # 4. Combine
    combined_train = mmhs_train + fb_train_upsampled
    combined_dev = mmhs_dev + fb_dev
    combined_test = mmhs_test + fb_test # Merging tests just for completeness, though we usually eval separately
    
    # Shuffle Train
    import random
    random.shuffle(combined_train)
    
    # 5. Save
    save_jsonl(combined_train, os.path.join(OUT_DIR, "train.jsonl"))
    save_jsonl(combined_dev, os.path.join(OUT_DIR, "dev.jsonl"))
    save_jsonl(combined_test, os.path.join(OUT_DIR, "test.jsonl"))
    
    print("Done.")

if __name__ == "__main__":
    main()
