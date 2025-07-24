# Azure OpenAI Integration

This document explains how to use Azure OpenAI with the ART Model class.

## Overview

The ART Model class now supports both standard OpenAI and Azure OpenAI configurations. The client type is automatically detected based on the parameters provided.

## Configuration

### Environment Variables

Create a `.env` file with your Azure OpenAI configuration:

```bash
# Azure OpenAI Configuration
AZURE_OPENAI_MODEL=gpt-4o-mini
AZURE_OPENAI_API_KEY=your-actual-azure-api-key
AZURE_OPENAI_BASE_URL=https://your-resource.openai.azure.com
AZURE_OPENAI_API_VERSION=2024-08-01-preview
AZURE_OPENAI_DEPLOYMENT=your-deployment-name

# Standard OpenAI Configuration (for comparison)
OPENAI_API_KEY=your-actual-openai-api-key
OPENAI_BASE_URL=https://api.openai.com/v1/
```

**⚠️ Security Note**: Never commit actual API keys to version control. Always use environment variables or secure secret management systems.

### Model Creation

#### Azure OpenAI Model

```python
import os
from art.model import Model

# Create Azure OpenAI model
azure_model = Model(
    name=os.getenv("AZURE_OPENAI_MODEL"),
    project="my-project",
    inference_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_BASE_URL"),
    azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION")
)
```

#### Standard OpenAI Model

```python
# Create standard OpenAI model
standard_model = Model(
    name="gpt-4",
    project="my-project",
    inference_api_key=os.getenv("OPENAI_API_KEY"),
    inference_base_url=os.getenv("OPENAI_BASE_URL")
)
```

## Key Parameters

### Azure OpenAI Specific Parameters

- `azure_endpoint`: Your Azure OpenAI resource endpoint (e.g., `https://your-resource.openai.azure.com`)
- `azure_deployment`: The deployment name you created in Azure OpenAI Studio
- `api_version`: The API version to use (defaults to `2024-02-01` if not specified)
- `inference_api_key`: Your Azure OpenAI API key

### Standard OpenAI Parameters

- `inference_base_url`: The OpenAI API base URL (e.g., `https://api.openai.com/v1/`)
- `inference_api_key`: Your OpenAI API key

## Client Detection

The Model class automatically detects which client to create:

- If `azure_endpoint` and `azure_deployment` are provided → Creates `AsyncAzureOpenAI` client
- Otherwise → Creates standard `AsyncOpenAI` client

## Usage Examples

### Basic Usage

```python
import asyncio
from art.model import Model

async def main():
    # Create Azure OpenAI model
    model = Model(
        name="gpt-4o-mini",
        project="test-project",
        inference_api_key="your-api-key",
        azure_endpoint="https://your-resource.openai.azure.com",
        azure_deployment="gpt-4o-mini-deployment",
        api_version="2024-08-01-preview"
    )
    
    # Get the OpenAI client (automatically creates Azure client)
    client = model.openai_client()
    
    # Use the client for completions
    # ... your completion logic here

asyncio.run(main())
```

### With Environment Variables

```python
import os
from dotenv import load_dotenv
from art.model import Model

# Load environment variables
load_dotenv()

# Create model using environment variables
model = Model(
    name=os.getenv("AZURE_OPENAI_MODEL"),
    project="my-project",
    inference_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_BASE_URL"),
    azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION")
)
```

## Testing

### Running Tests

```bash
# Install test dependencies
uv add --dev pytest python-dotenv

# Run the Azure OpenAI tests
python run_azure_tests.py

# Or run with pytest directly
pytest src/art/test/test_azure_openai_model.py -v
```

### Test Configuration

The tests use a `.env.test` file with test configuration. Make sure to update it with your actual test credentials if needed.

### Example Script

Run the example script to see the integration in action:

```bash
python examples/azure_openai_example.py
```

## Error Handling

The Model class provides clear error messages for common configuration issues:

- Missing Azure endpoint or deployment
- Missing API key
- Missing base URL for standard OpenAI

## LiteLLM Integration

The Model class also provides `litellm_completion_params()` method that returns parameters compatible with LiteLLM for both Azure and standard OpenAI configurations.

## Notes

- The `api_version` parameter defaults to `"2024-02-01"` if not specified for Azure OpenAI
- Both client types support the same timeout and connection limit configurations
- The `patch_openai` function is applied to both client types for consistent behavior
- Client instances are cached after first creation for performance