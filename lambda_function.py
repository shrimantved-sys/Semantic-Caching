"""AWS Lambda Entrypoint for Two-Tier Semantic LLM Cache & Inference Engine.

Wraps the FastAPI application using Mangum for AWS Lambda Function URL and API Gateway.
Region: ap-southeast-2
"""

import os
from mangum import Mangum
from server import app

# Lambda handler with lifespan='off' for fast cold-start performance
handler = Mangum(app, lifespan="off")
