#!/usr/bin/env python3
"""
Download marker-pdf model for PDF processing
"""
from modelscope import snapshot_download

def download_model():
    """Download the marker-pdf model"""
    print("Downloading marker-pdf model...")
    snapshot_download('Lixiang/marker-pdf', local_dir='models')
    print("Model download completed!")

if __name__ == "__main__":
    download_model()
