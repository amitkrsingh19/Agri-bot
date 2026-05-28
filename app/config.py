import os
from dotenv import load_dotenv

# Load environment variables from a local .env file (development) if present.
load_dotenv(override=True)

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")


if not GOOGLE_API_KEY:
    raise ValueError(
        "❌ GOOGLE_API_KEY is missing! Add it to .env or set it in your environment variables."
    )
