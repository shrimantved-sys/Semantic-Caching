#!/usr/bin/env python3
"""Automated AWS Lambda Deployment Script for Two-Tier Semantic LLM Cache.

Region: ap-southeast-2 (Strictly within selected Region)
Resources Provisioned:
1. Amazon DynamoDB Tables: 'llm-semantic-cache' (with TTL), 'llm-telemetry-logs'
2. IAM Execution Role: 'LLMCacheLambdaExecutionRole'
3. AWS Lambda Function: 'llm-semantic-cache-service' (Python 3.12, x86_64)
4. AWS Lambda Function URL: Public HTTPS endpoint with CORS enabled
"""

import io
import json
import os
import shutil
import subprocess
import sys
import time
import zipfile

import boto3
from botocore.exceptions import ClientError
from dotenv import load_dotenv

load_dotenv()

# Configuration
REGION = os.environ.get("AWS_REGION", "ap-southeast-2")
FUNCTION_NAME = "llm-semantic-cache-service"
ROLE_NAME = "LLMCacheLambdaExecutionRole"
CACHE_TABLE = "llm-semantic-cache"
TELEMETRY_TABLE = "llm-telemetry-logs"
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "").strip()
GROQ_MODEL_NAME = os.environ.get("GROQ_MODEL_NAME", "qwen/qwen3.8-27b")
BEDROCK_EMBED_MODEL = os.environ.get("BEDROCK_EMBED_MODEL", "amazon.titan-embed-text-v2:0")

PKG_DIR = "lambda_pkg"
ZIP_FILE = "lambda_deployment.zip"


def log(msg: str):
    print(f"[Deploy] {msg}")


def get_account_id() -> str:
    sts = boto3.client("sts", region_name=REGION)
    return sts.get_caller_identity()["Account"]


def ensure_dynamodb_tables():
    log(f"Verifying DynamoDB tables in selected Region ({REGION})...")
    ddb = boto3.client("dynamodb", region_name=REGION)

    # 1. Cache Table
    try:
        ddb.create_table(
            TableName=CACHE_TABLE,
            KeySchema=[{"AttributeName": "cache_key", "KeyType": "HASH"}],
            AttributeDefinitions=[{"AttributeName": "cache_key", "AttributeType": "S"}],
            BillingMode="PAY_PER_REQUEST"
        )
        log(f"Created DynamoDB table: {CACHE_TABLE}")
    except ddb.exceptions.ResourceInUseException:
        log(f"DynamoDB table '{CACHE_TABLE}' already exists.")

    # 2. Telemetry Table
    try:
        ddb.create_table(
            TableName=TELEMETRY_TABLE,
            KeySchema=[{"AttributeName": "request_id", "KeyType": "HASH"}],
            AttributeDefinitions=[{"AttributeName": "request_id", "AttributeType": "S"}],
            BillingMode="PAY_PER_REQUEST"
        )
        log(f"Created DynamoDB table: {TELEMETRY_TABLE}")
    except ddb.exceptions.ResourceInUseException:
        log(f"DynamoDB table '{TELEMETRY_TABLE}' already exists.")

    # Enable TTL on cache table
    waiter = ddb.get_waiter("table_exists")
    waiter.wait(TableName=CACHE_TABLE)
    try:
        ddb.update_time_to_live(
            TableName=CACHE_TABLE,
            TimeToLiveSpecification={"Enabled": True, "AttributeName": "ttl_timestamp"}
        )
        log(f"TTL verified on table: {CACHE_TABLE} (attribute: ttl_timestamp)")
    except Exception as e:
        log(f"TTL status note: {e}")


def ensure_iam_role(account_id: str) -> str:
    log("Verifying IAM execution role...")
    iam = boto3.client("iam")
    trust_policy = {
        "Version": "2012-10-17",
        "Statement": [{
            "Effect": "Allow",
            "Principal": {"Service": "lambda.amazonaws.com"},
            "Action": "sts:AssumeRole"
        }]
    }

    role_arn = ""
    try:
        resp = iam.create_role(
            RoleName=ROLE_NAME,
            AssumeRolePolicyDocument=json.dumps(trust_policy),
            Description="Execution role for LLM Two-Tier Semantic Cache Lambda"
        )
        role_arn = resp["Role"]["Arn"]
        log(f"Created IAM role: {ROLE_NAME}")
        # Wait a moment for IAM propagation
        time.sleep(8)
    except iam.exceptions.EntityAlreadyExistsException:
        role_arn = iam.get_role(RoleName=ROLE_NAME)["Role"]["Arn"]
        log(f"IAM role '{ROLE_NAME}' exists: {role_arn}")

    # Attach basic execution policy
    iam.attach_role_policy(
        RoleName=ROLE_NAME,
        PolicyArn="arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"
    )

    # Attach inline policy for DynamoDB and Bedrock
    inline_policy = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Action": [
                    "dynamodb:*"
                ],
                "Resource": f"arn:aws:dynamodb:{REGION}:{account_id}:table/llm-*"
            },
            {
                "Effect": "Allow",
                "Action": ["bedrock:InvokeModel"],
                "Resource": "*"
            }
        ]
    }
    iam.put_role_policy(
        RoleName=ROLE_NAME,
        PolicyName="LLMCacheDynamoDBAndBedrockPolicy",
        PolicyDocument=json.dumps(inline_policy)
    )
    log("IAM policies verified.")
    return role_arn


def build_package() -> str:
    log("Building Lambda deployment zip package...")

    # Ensure pip dependencies installed in lambda_pkg
    if not os.path.exists(PKG_DIR):
        log(f"Installing dependencies into {PKG_DIR}...")
        cmd = [
            sys.executable, "-m", "pip", "install",
            "-t", PKG_DIR,
            "--platform", "manylinux2014_x86_64",
            "--only-binary=:all:",
            "--python-version", "3.12",
            "--upgrade",
            "mangum", "fastapi", "httpx", "pydantic", "pydantic-core", "python-dotenv"
        ]
        subprocess.check_call(cmd)

    # Copy project application files into PKG_DIR
    files_to_copy = ["server.py", "aws_storage.py", "lambda_function.py", "index.html"]
    for fn in files_to_copy:
        if os.path.exists(fn):
            shutil.copy(fn, os.path.join(PKG_DIR, fn))

    # Create zip file
    log(f"Creating zip file: {ZIP_FILE}...")
    with zipfile.ZipFile(ZIP_FILE, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, dirs, files in os.walk(PKG_DIR):
            for file in files:
                full_path = os.path.join(root, file)
                rel_path = os.path.relpath(full_path, PKG_DIR)
                zf.write(full_path, rel_path)

    zip_size_mb = os.path.getsize(ZIP_FILE) / (1024 * 1024)
    log(f"Package built successfully! Size: {zip_size_mb:.2f} MB")
    return ZIP_FILE


def deploy_lambda_function(role_arn: str, zip_path: str) -> str:
    log(f"Deploying Lambda function '{FUNCTION_NAME}' to region {REGION}...")
    lam = boto3.client("lambda", region_name=REGION)

    with open(zip_path, "rb") as f:
        zip_bytes = f.read()

    env_vars = {
        "GROQ_API_KEY": GROQ_API_KEY,
        "GROQ_MODEL_NAME": GROQ_MODEL_NAME,
        "CACHE_TABLE_NAME": CACHE_TABLE,
        "TELEMETRY_TABLE_NAME": TELEMETRY_TABLE,
        "BEDROCK_EMBED_MODEL": BEDROCK_EMBED_MODEL
    }


    try:
        # Check if function exists
        lam.get_function(FunctionName=FUNCTION_NAME)
        log("Function exists. Updating code and configuration...")

        lam.update_function_code(
            FunctionName=FUNCTION_NAME,
            ZipFile=zip_bytes
        )

        # Wait for update to complete
        waiter = lam.get_waiter("function_updated")
        waiter.wait(FunctionName=FUNCTION_NAME)

        lam.update_function_configuration(
            FunctionName=FUNCTION_NAME,
            Role=role_arn,
            Handler="lambda_function.handler",
            Runtime="python3.12",
            Timeout=30,
            MemorySize=512,
            Environment={"Variables": env_vars}
        )
        waiter.wait(FunctionName=FUNCTION_NAME)
        log("Function updated successfully.")

    except lam.exceptions.ResourceNotFoundException:
        log("Creating new Lambda function...")
        # IAM role sometimes needs a few seconds after creation
        for attempt in range(5):
            try:
                lam.create_function(
                    FunctionName=FUNCTION_NAME,
                    Runtime="python3.12",
                    Role=role_arn,
                    Handler="lambda_function.handler",
                    Code={"ZipFile": zip_bytes},
                    Description="Two-Tier Semantic LLM Cache & Inference Engine",
                    Timeout=30,
                    MemorySize=512,
                    Publish=True,
                    Environment={"Variables": env_vars}
                )
                break
            except ClientError as err:
                if "The role defined for the function cannot be assumed" in str(err) and attempt < 4:
                    log("Waiting for IAM role propagation...")
                    time.sleep(5)
                else:
                    raise

        waiter = lam.get_waiter("function_active")
        waiter.wait(FunctionName=FUNCTION_NAME)
        log("Function created successfully.")

    # Configure Function URL (AuthType: NONE)
    log("Configuring Lambda Function URL...")
    function_url = ""
    try:
        url_resp = lam.get_function_url_config(FunctionName=FUNCTION_NAME)
        function_url = url_resp["FunctionUrl"]
        log(f"Existing Function URL found: {function_url}")
    except lam.exceptions.ResourceNotFoundException:
        url_resp = lam.create_function_url_config(
            FunctionName=FUNCTION_NAME,
            AuthType="NONE",
            Cors={
                "AllowOrigins": ["*"],
                "AllowMethods": ["*"],
                "AllowHeaders": ["*"]
            }
        )
        function_url = url_resp["FunctionUrl"]
        log(f"Created Function URL: {function_url}")

    # Add public invocation permission for Function URL
    try:
        lam.add_permission(
            FunctionName=FUNCTION_NAME,
            StatementId="FunctionURLAllowPublicAccess",
            Action="lambda:InvokeFunctionUrl",
            Principal="*",
            FunctionUrlAuthType="NONE"
        )
        log("Public InvokeFunctionUrl permission granted.")
    except lam.exceptions.ResourceConflictException:
        pass

    try:
        lam.add_permission(
            FunctionName=FUNCTION_NAME,
            StatementId="FunctionURLAllowInvokeFunction",
            Action="lambda:InvokeFunction",
            Principal="*"
        )
        log("Public InvokeFunction permission granted.")
    except lam.exceptions.ResourceConflictException:
        pass

    return function_url


def main():
    log("=" * 60)
    log("Deploying Two-Tier Semantic LLM Cache to AWS Lambda")
    log(f"Selected Region: {REGION}")
    log("=" * 60)

    account_id = get_account_id()
    log(f"AWS Account ID: {account_id}")

    ensure_dynamodb_tables()
    role_arn = ensure_iam_role(account_id)
    zip_path = build_package()
    function_url = deploy_lambda_function(role_arn, zip_path)

    log("=" * 60)
    log("Deployment Complete!")
    log(f"Live Endpoint: {function_url}")
    log(f"Playground UI: {function_url}playground")
    log(f"Status API:    {function_url}api/status")
    log("=" * 60)
    print(f"\nLAMBDA_FUNCTION_URL={function_url}")


if __name__ == "__main__":
    main()
