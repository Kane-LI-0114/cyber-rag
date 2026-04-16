"""Test LLM API connectivity using project .env configuration."""

from __future__ import annotations

import sys

from cyber_rag.config import GenerationConfig
from langchain_openai import AzureChatOpenAI, ChatOpenAI


def _mask_secret(secret: str | None) -> str:
    if not secret:
        return "NOT SET"
    if len(secret) <= 8:
        return "*" * len(secret)
    return f"{secret[:4]}...{secret[-4:]}"


def main() -> None:
    config = GenerationConfig()
    print("=" * 60)
    print("CyberRAG LLM Connection Test")
    print("=" * 60)
    print(f"Provider : {config.provider}")
    print(f"Model    : {config.model_name or 'NOT SET'}")
    print(f"Base URL : {config.base_url or 'NOT SET'}")
    print(f"API Key  : {_mask_secret(config.api_key)}")
    if config.provider == "azure":
        print(f"Version  : {config.api_version or 'NOT SET'}")
    print("-" * 60)

    if not config.is_configured:
        print("FAILED: Missing required configuration in .env")
        print("Please check provider-specific key/base_url/model settings.")
        sys.exit(1)

    try:
        if config.provider == "azure":
            llm = AzureChatOpenAI(
                azure_deployment=config.model_name or "gpt-4o-mini",
                api_version=config.api_version or "2024-10-21",
                azure_endpoint=config.base_url.rstrip("/"),
                api_key=config.api_key,
                temperature=0.0,
                max_retries=1,
            )
        elif config.provider == "oneapi":
            llm = ChatOpenAI(
                model=config.model_name or "gpt-3.5-turbo",
                base_url=config.base_url.rstrip("/"),
                api_key=config.api_key,
                temperature=0.0,
                max_retries=1,
            )
        else:
            print(f"FAILED: Unsupported provider '{config.provider}'")
            sys.exit(1)

        response = llm.invoke("Reply with exactly: CONNECTION_OK")
        content = str(response.content).strip()
        print("SUCCESS: API connection works.")
        print(f"Model response: {content}")
        sys.exit(0)
    except Exception as exc:  # noqa: BLE001 - show full provider error details
        print(f"FAILED: {type(exc).__name__}")
        print(str(exc))
        if "401" in str(exc) or "invalid subscription key" in str(exc).lower():
            print("Hint: API key is invalid, expired, or mismatched with the endpoint.")
        elif "404" in str(exc):
            print("Hint: Model/deployment name may be wrong for this endpoint.")
        elif "429" in str(exc):
            print("Hint: Rate limited. Retry later or use an authenticated higher quota.")
        sys.exit(2)


if __name__ == "__main__":
    main()
