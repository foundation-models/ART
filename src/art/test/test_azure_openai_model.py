#!/usr/bin/env python3
"""
Unit tests for Azure OpenAI integration in the Model class.
Tests reading environment variables and creating Azure OpenAI clients.
"""

import os
import pytest
from unittest.mock import patch, MagicMock
from dotenv import load_dotenv
from pathlib import Path

# Import the Model class
from art.model import Model


class TestAzureOpenAIModel:
    """Test suite for Azure OpenAI Model integration."""
    
    @classmethod
    def setup_class(cls):
        """Load test environment variables from .env.test file."""
        # Load the test .env file
        test_env_path = Path(__file__).parent.parent.parent.parent / ".env.test"
        if test_env_path.exists():
            load_dotenv(test_env_path)
    
    def test_load_env_variables(self):
        """Test that all required environment variables are loaded correctly."""
        # Test Azure OpenAI environment variables exist
        assert os.getenv("AZURE_OPENAI_MODEL") is not None
        assert os.getenv("AZURE_OPENAI_API_KEY") is not None
        assert os.getenv("AZURE_OPENAI_BASE_URL") is not None
        assert os.getenv("AZURE_OPENAI_API_VERSION") is not None
        assert os.getenv("AZURE_OPENAI_DEPLOYMENT") is not None
        
        # Test standard OpenAI environment variables exist
        assert os.getenv("OPENAI_API_KEY") is not None
        assert os.getenv("OPENAI_BASE_URL") is not None
    
    def test_create_azure_openai_model_from_env(self):
        """Test creating an Azure OpenAI model using environment variables."""
        model = Model(
            name=os.getenv("AZURE_OPENAI_MODEL", "gpt-4o-mini"),
            project="test-project",
            inference_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            azure_endpoint=os.getenv("AZURE_OPENAI_BASE_URL"),
            azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION")
        )
        
        # Verify model properties using environment variables
        assert model.name == os.getenv("AZURE_OPENAI_MODEL")
        assert model.project == "test-project"
        assert model.inference_api_key == os.getenv("AZURE_OPENAI_API_KEY")
        assert model.azure_endpoint == os.getenv("AZURE_OPENAI_BASE_URL")
        assert model.azure_deployment == os.getenv("AZURE_OPENAI_DEPLOYMENT")
        assert model.api_version == os.getenv("AZURE_OPENAI_API_VERSION")
        assert not model.trainable
    
    def test_create_standard_openai_model_from_env(self):
        """Test creating a standard OpenAI model using environment variables."""
        model = Model(
            name="gpt-4",
            project="test-project",
            inference_api_key=os.getenv("OPENAI_API_KEY"),
            inference_base_url=os.getenv("OPENAI_BASE_URL")
        )
        
        # Verify model properties using environment variables
        assert model.name == "gpt-4"
        assert model.project == "test-project"
        assert model.inference_api_key == os.getenv("OPENAI_API_KEY")
        assert model.inference_base_url == os.getenv("OPENAI_BASE_URL")
        assert model.azure_endpoint is None
        assert model.azure_deployment is None
        assert model.api_version is None
        assert not model.trainable
    
    @patch('art.model.AsyncAzureOpenAI')
    @patch('art.model.patch_openai')
    def test_azure_openai_client_creation(self, mock_patch_openai, mock_azure_client):
        """Test that Azure OpenAI client is created correctly."""
        # Setup mock
        mock_client_instance = MagicMock()
        mock_azure_client.return_value = mock_client_instance
        
        # Create model with Azure configuration
        model = Model(
            name=os.getenv("AZURE_OPENAI_MODEL", "gpt-4o-mini"),
            project="test-project",
            inference_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            azure_endpoint=os.getenv("AZURE_OPENAI_BASE_URL"),
            azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION")
        )
        
        # Get the client
        client = model.openai_client()
        
        # Verify Azure client was created with correct parameters
        mock_azure_client.assert_called_once()
        call_kwargs = mock_azure_client.call_args[1]
        
        assert call_kwargs['azure_endpoint'] == os.getenv("AZURE_OPENAI_BASE_URL")
        assert call_kwargs['azure_deployment'] == os.getenv("AZURE_OPENAI_DEPLOYMENT")
        assert call_kwargs['api_key'] == os.getenv("AZURE_OPENAI_API_KEY")
        assert call_kwargs['api_version'] == os.getenv("AZURE_OPENAI_API_VERSION")
        assert 'http_client' in call_kwargs
        
        # Verify patch_openai was called
        mock_patch_openai.assert_called_once_with(mock_client_instance)
        
        # Verify the client is cached
        assert model._openai_client == mock_client_instance
        assert client == mock_client_instance
    
    @patch('art.model.AsyncOpenAI')
    @patch('art.model.patch_openai')
    def test_standard_openai_client_creation(self, mock_patch_openai, mock_openai_client):
        """Test that standard OpenAI client is created correctly."""
        # Setup mock
        mock_client_instance = MagicMock()
        mock_openai_client.return_value = mock_client_instance
        
        # Create model with standard OpenAI configuration
        model = Model(
            name="gpt-4",
            project="test-project",
            inference_api_key=os.getenv("OPENAI_API_KEY"),
            inference_base_url=os.getenv("OPENAI_BASE_URL")
        )
        
        # Get the client
        client = model.openai_client()
        
        # Verify standard OpenAI client was created with correct parameters
        mock_openai_client.assert_called_once()
        call_kwargs = mock_openai_client.call_args[1]
        
        assert call_kwargs['base_url'] == os.getenv("OPENAI_BASE_URL")
        assert call_kwargs['api_key'] == os.getenv("OPENAI_API_KEY")
        assert 'http_client' in call_kwargs
        
        # Verify patch_openai was called
        mock_patch_openai.assert_called_once_with(mock_client_instance)
        
        # Verify the client is cached
        assert model._openai_client == mock_client_instance
        assert client == mock_client_instance
    
    def test_azure_client_missing_endpoint_error(self):
        """Test error handling when Azure endpoint is missing."""
        model = Model(
            name="gpt-4o-mini",
            project="test-project",
            inference_api_key="test-key",
            azure_deployment="test-deployment"
            # Missing azure_endpoint
        )
        
        with pytest.raises(ValueError, match="In order to create an OpenAI client you must provide"):
            model.openai_client()
    
    def test_azure_client_missing_deployment_error(self):
        """Test error handling when Azure deployment is missing."""
        model = Model(
            name="gpt-4o-mini",
            project="test-project",
            inference_api_key="test-key",
            azure_endpoint="https://test.openai.azure.com"
            # Missing azure_deployment
        )
        
        with pytest.raises(ValueError, match="In order to create an OpenAI client you must provide"):
            model.openai_client()
    
    def test_azure_client_missing_api_key_error(self):
        """Test error handling when API key is missing for Azure."""
        model = Model(
            name="gpt-4o-mini",
            project="test-project",
            azure_endpoint="https://test.openai.azure.com",
            azure_deployment="test-deployment"
            # Missing inference_api_key
        )
        
        with pytest.raises(ValueError, match="In order to create an Azure OpenAI client you must provide an `inference_api_key`"):
            model.openai_client()
    
    def test_standard_client_missing_base_url_error(self):
        """Test error handling when base URL is missing for standard OpenAI."""
        model = Model(
            name="gpt-4",
            project="test-project",
            inference_api_key="test-key"
            # Missing inference_base_url
        )
        
        with pytest.raises(ValueError, match="In order to create an OpenAI client you must provide an `inference_api_key` and `inference_base_url`"):
            model.openai_client()
    
    def test_litellm_completion_params_azure(self):
        """Test litellm completion parameters for Azure model."""
        model = Model(
            name=os.getenv("AZURE_OPENAI_MODEL", "gpt-4o-mini"),
            project="test-project",
            inference_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            azure_endpoint=os.getenv("AZURE_OPENAI_BASE_URL"),
            azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
            inference_model_name="custom-model-name"
        )
        
        params = model.litellm_completion_params()
        
        assert params["model"] == "custom-model-name"
        assert params["base_url"] == os.getenv("AZURE_OPENAI_BASE_URL")
        assert params["api_key"] == os.getenv("AZURE_OPENAI_API_KEY")
        assert params["temperature"] == 1
    
    def test_get_inference_name_with_custom_name(self):
        """Test get_inference_name with custom inference_model_name."""
        model = Model(
            name="gpt-4o-mini",
            project="test-project",
            inference_api_key="test-key",
            azure_endpoint="https://test.openai.azure.com",
            azure_deployment="test-deployment",
            inference_model_name="custom-deployment-name"
        )
        
        assert model.get_inference_name() == "custom-deployment-name"
    
    def test_get_inference_name_fallback_to_name(self):
        """Test get_inference_name falls back to model name."""
        model = Model(
            name="gpt-4o-mini",
            project="test-project",
            inference_api_key="test-key",
            azure_endpoint="https://test.openai.azure.com",
            azure_deployment="test-deployment"
        )
        
        assert model.get_inference_name() == "gpt-4o-mini"


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v"])