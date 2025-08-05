"""
Hugging Face Spaces entry point.
This file is required for Hugging Face Spaces deployment.
"""

import logging
import os
import sys

import uvicorn
from dotenv import load_dotenv

load_dotenv()


# Hugging Face Spaces will automatically run this app
if __name__ == "__main__":
    # Check for the --gen-creds flag
    if "--gen-creds" in sys.argv:
        print("Starting the credential generator...")
        # Import the generator app and run it
        from src.generate_credentials import app
        from src.settings import settings

        print(f"Please open your web browser and navigate to {settings.DOMAIN_NAME}")
        uvicorn.run(app, host="0.0.0.0", port=settings.PORT)
    else:
        print("Starting the Gemini API proxy...")
        # Import the main proxy app and run it
        from src.main import app
        from src.settings import settings

        uvicorn.run(app, host="0.0.0.0", port=settings.PORT)
