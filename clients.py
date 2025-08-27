from slack_sdk.web.async_client import AsyncWebClient
from slack_sdk.signature import SignatureVerifier
from fastapi import FastAPI
from openai import AsyncOpenAI
from config import SLACK_BOT_TOKEN, SLACK_SIGNING_SECRET, OPENAI_API_KEY
from logger import logger

# Initialize Slack client
slack_client = AsyncWebClient(token=SLACK_BOT_TOKEN)
signature_verifier = SignatureVerifier(SLACK_SIGNING_SECRET)

# Initialize FastAPI app
api = FastAPI(title="Design request API", version="2.0")

# Initialize OpenAI client
client = AsyncOpenAI(api_key=OPENAI_API_KEY)

# Validate required environment variables
if not SLACK_BOT_TOKEN:
    logger.error("SLACK_BOT_TOKEN not found in environment variables")
    raise ValueError("SLACK_BOT_TOKEN is required")
if not SLACK_SIGNING_SECRET:
    logger.error("SLACK_SIGNING_SECRET not found in environment variables")
    raise ValueError("SLACK_SIGNING_SECRET is required")
if not OPENAI_API_KEY:
    logger.error("OPENAI_API_KEY not found in environment variables")
    raise ValueError("OPENAI_API_KEY is required")