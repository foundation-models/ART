#!/usr/bin/env python3
"""
Example demonstrating Azure OpenAI integration with ART Model class.
This example shows how to create and use Azure OpenAI models with environment variables.
"""

import os
import asyncio
from dotenv import load_dotenv
from pathlib import Path

# Import the Model class
from art.model import Model


async def main():
    """Demonstrate Azure OpenAI integration."""
    print("🚀 Azure OpenAI Integration Example")
    print("=" * 40)
    
    # Load environment variables from .env.test file
    env_file = Path(__file__).parent.parent / ".env.test"
    if env_file.exists():
        load_dotenv(env_file)
        print(f"✅ Loaded environment variables from {env_file}")
    else:
        print("⚠️  No .env.test file found, using system environment variables")
    
    # Create Azure OpenAI model using environment variables
    print("\n📝 Creating Azure OpenAI Model...")
    azure_model = Model(
        name=os.getenv("AZURE_OPENAI_MODEL", "gpt-4o-mini"),
        project="azure-example-project",
        inference_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        azure_endpoint=os.getenv("AZURE_OPENAI_BASE_URL"),
        azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION")
    )
    
    print(f"✅ Azure Model Created:")
    print(f"   Name: {azure_model.name}")
    print(f"   Project: {azure_model.project}")
    print(f"   Azure Endpoint: {azure_model.azure_endpoint}")
    print(f"   Azure Deployment: {azure_model.azure_deployment}")
    print(f"   API Version: {azure_model.api_version}")
    print(f"   API Key: {'***' + (azure_model.inference_api_key[-4:] if azure_model.inference_api_key else 'None')}")
    print(f"   Trainable: {azure_model.trainable}")
    
    # Create standard OpenAI model for comparison
    print("\n📝 Creating Standard OpenAI Model...")
    standard_model = Model(
        name="gpt-4",
        project="standard-example-project",
        inference_api_key=os.getenv("OPENAI_API_KEY"),
        inference_base_url=os.getenv("OPENAI_BASE_URL")
    )
    
    print(f"✅ Standard Model Created:")
    print(f"   Name: {standard_model.name}")
    print(f"   Project: {standard_model.project}")
    print(f"   Base URL: {standard_model.inference_base_url}")
    print(f"   API Key: {'***' + (standard_model.inference_api_key[-4:] if standard_model.inference_api_key else 'None')}")
    print(f"   Trainable: {standard_model.trainable}")
    
    # Demonstrate client creation (without actually calling the API)
    print("\n🔧 Testing Client Creation...")
    
    try:
        # This will create the Azure OpenAI client
        azure_client = azure_model.openai_client()
        print(f"✅ Azure OpenAI client created: {type(azure_client).__name__}")
        
        # This will create the standard OpenAI client
        standard_client = standard_model.openai_client()
        print(f"✅ Standard OpenAI client created: {type(standard_client).__name__}")
        
    except Exception as e:
        print(f"⚠️  Client creation test (expected with test credentials): {e}")
    
    # Show litellm completion parameters
    print("\n⚙️  LiteLLM Completion Parameters:")
    azure_params = azure_model.litellm_completion_params()
    print(f"Azure Model: {azure_params}")
    
    standard_params = standard_model.litellm_completion_params()
    print(f"Standard Model: {standard_params}")
    
    # Show inference names
    print("\n🏷️  Inference Names:")
    print(f"Azure Model: {azure_model.get_inference_name()}")
    print(f"Standard Model: {standard_model.get_inference_name()}")
    
    print("\n✨ Example completed successfully!")


if __name__ == "__main__":
    asyncio.run(main())