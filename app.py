"""
Hugging Face Spaces entry point.
This file is required for Hugging Face Spaces deployment.
"""

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
        from src.core.settings import settings
        from src.tools.generate_credentials import app

        print(f"Please open your web browser and navigate to {settings.DOMAIN_NAME}")
        uvicorn.run(app, host="0.0.0.0", port=settings.PORT)
    else:
        print("Starting the Gemini API proxy...")
        # Import the main proxy app and run it
        from src.core.settings import settings
        from src.main import app

        uvicorn.run(app, host="0.0.0.0", port=settings.PORT)
