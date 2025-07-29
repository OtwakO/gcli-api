"""
Hugging Face Spaces entry point.
This file is required for Hugging Face Spaces deployment.
"""

import logging
import os

import uvicorn
from dotenv import load_dotenv

load_dotenv()

# from src.main import app
from src.main import app

# Hugging Face Spaces will automatically run this app
if __name__ == "__main__":
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "7860"))
    logging.info(f"{os.getenv('OAUTH_CREDS_JSON')}")
    uvicorn.run(app, host=host, port=port)
