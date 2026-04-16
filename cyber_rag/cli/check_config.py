"""Check the current configuration status."""

import sys


def main() -> None:
    """Check and display the current CyberRAG configuration."""
    from cyber_rag.config import GenerationConfig, print_config_status

    print("=" * 60)
    print("CyberRAG Configuration Status")
    print("=" * 60)
    print()

    print_config_status()
    print()

    # Validate configuration
    config = GenerationConfig()
    if not config.is_configured:
        print("WARNING: Current provider is not fully configured!")
        print()
        print("To fix this, edit .env file:")
        print(f"  1. Set CYBER_RAG_LLM_PROVIDER={config.provider}")
        print("  2. Fill in the corresponding API credentials")
        print()
        print("Available providers: azure, oneapi")
        print()
        print("Example .env configurations:")
        print("-" * 40)
        print("# For Azure:")
        print("CYBER_RAG_LLM_PROVIDER=azure")
        print("CYBER_RAG_AZURE_API_KEY=your_key")
        print("CYBER_RAG_AZURE_BASE_URL=https://xxx.openai.azure.com/openai")
        print("CYBER_RAG_AZURE_MODEL_NAME=gpt-4o-mini")
        print()
        print("# For OneAPI:")
        print("CYBER_RAG_LLM_PROVIDER=oneapi")
        print("CYBER_RAG_ONEAPI_API_KEY=your_key")
        print("CYBER_RAG_ONEAPI_BASE_URL=https://api.example.com/v1")
        print("CYBER_RAG_ONEAPI_MODEL_NAME=deepseek-v3")
        sys.exit(1)
    else:
        print("Configuration is valid!")
        print()
        print("To switch provider, change CYBER_RAG_LLM_PROVIDER in .env:")
        print("  - Set to 'azure' for Azure OpenAI")
        print("  - Set to 'oneapi' for OneAPI/OpenAI-compatible API")
    print()
    print("=" * 60)


if __name__ == "__main__":
    main()
