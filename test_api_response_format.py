import argparse
import getpass
import json

from openai import OpenAI

from llm import LLM


def ask(prompt: str, default: str | None = None, secret: bool = False) -> str:
    suffix = f" [{default}]" if default else ""
    full_prompt = f"{prompt}{suffix}: "
    value = getpass.getpass(full_prompt) if secret else input(full_prompt).strip()
    if not value and default is not None:
        return default
    return value


def to_jsonable(obj):
    if obj is None:
        return None
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if isinstance(obj, dict):
        return obj
    return str(obj)


def main():
    parser = argparse.ArgumentParser(
        description="Interactively test chat completion response format (reasoning vs content)."
    )
    parser.add_argument("--base-url", type=str, default=None)
    parser.add_argument("--api-key", type=str, default=None)
    parser.add_argument("--model", type=str, default="gpt-4o")
    parser.add_argument("--thinking", action="store_true", help="Enable thinking mode in request body.")
    parser.add_argument(
        "--prompt",
        type=str,
        default="Summarize why transformers work in one sentence.",
    )
    args = parser.parse_args()

    base_url = args.base_url or ask("API base URL", "https://api.openai.com/v1")
    api_key = args.api_key or ask("API key", secret=True)
    model = args.model or ask("Model name", "gpt-4o")

    if not args.api_key:
        print("\nAPI key was entered interactively.")
    print(f"Testing model: {model}")
    print(f"Base URL: {base_url}")
    print(f"Thinking enabled: {args.thinking}\n")

    client = OpenAI(api_key=api_key, base_url=base_url)
    request_kwargs = {
        "model": model,
        "temperature": 0,
        "messages": [
            {"role": "system", "content": "You are a concise assistant."},
            {"role": "user", "content": args.prompt},
        ],
    }
    if args.thinking:
        request_kwargs["extra_body"] = {"enable_thinking": True}

    response = client.chat.completions.create(**request_kwargs)
    message = response.choices[0].message

    print("=== RAW MESSAGE (JSON) ===")
    print(json.dumps(to_jsonable(message), ensure_ascii=False, indent=2))

    print("\n=== DIRECT FIELDS ===")
    print("message.content:")
    print(message.content)
    print("\nmessage.reasoning_content:")
    print(getattr(message, "reasoning_content", None))

    llm = LLM(api_key=api_key, base_url=base_url, model=model, thinking=args.thinking)
    parsed = llm.generate(
        messages=[
            {"role": "system", "content": "You are a concise assistant."},
            {"role": "user", "content": args.prompt},
        ]
    )

    print("\n=== PARSED BY PROJECT LLM ===")
    print("parsed.reasoning_content:")
    print(parsed.reasoning_content)
    print("\nparsed.content:")
    print(parsed.content)


if __name__ == "__main__":
    main()
