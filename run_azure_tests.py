#!/usr/bin/env python3
"""
Test runner for Azure OpenAI integration tests.
This script runs the Azure OpenAI model tests and displays the results.
"""

import subprocess
import sys
from pathlib import Path

def main():
    """Run the Azure OpenAI tests."""
    print("🧪 Running Azure OpenAI Model Tests...")
    print("=" * 50)
    
    # Get the test file path
    test_file = Path(__file__).parent / "src" / "art" / "test" / "test_azure_openai_model.py"
    
    if not test_file.exists():
        print(f"❌ Test file not found: {test_file}")
        sys.exit(1)
    
    try:
        # Run pytest with verbose output
        result = subprocess.run([
            sys.executable, "-m", "pytest", 
            str(test_file), 
            "-v", 
            "--tb=short",
            "--color=yes"
        ], cwd=Path(__file__).parent)
        
        if result.returncode == 0:
            print("\n✅ All tests passed!")
        else:
            print(f"\n❌ Tests failed with return code: {result.returncode}")
            
        return result.returncode
        
    except FileNotFoundError:
        print("❌ pytest not found. Please install it with: uv add --dev pytest")
        return 1
    except Exception as e:
        print(f"❌ Error running tests: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())