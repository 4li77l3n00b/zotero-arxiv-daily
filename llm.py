from llama_cpp import Llama
from openai import OpenAI
from loguru import logger
from time import sleep
from dataclasses import dataclass
from typing import Optional

GLOBAL_LLM = None


@dataclass
class LLMGenerateResult:
    content: str
    reasoning_content: Optional[str] = None

class LLM:
    def __init__(self, api_key: str = None, base_url: str = None, model: str = None,lang: str = "English", thinking: bool = False):
        if api_key:
            self.llm = OpenAI(api_key=api_key, base_url=base_url)
        else:
            self.llm = Llama.from_pretrained(
                repo_id="Qwen/Qwen2.5-3B-Instruct-GGUF",
                filename="qwen2.5-3b-instruct-q4_k_m.gguf",
                n_ctx=5_000,
                n_threads=4,
                verbose=False,
            )
        self.model = model
        self.lang = lang
        self.thinking = thinking

    @staticmethod
    def _normalize_text(text) -> str:
        if text is None:
            return ""
        if isinstance(text, str):
            return text
        if isinstance(text, list):
            parts = []
            for item in text:
                if isinstance(item, dict):
                    parts.append(item.get("text", ""))
                else:
                    parts.append(str(item))
            return "".join(parts)
        return str(text)

    @staticmethod
    def _split_think_tag(content: str) -> tuple[Optional[str], str]:
        if "</think>" not in content:
            return None, content
        think_part, answer_part = content.split("</think>", 1)
        think_part = think_part.replace("<think>", "").strip()
        return think_part if think_part else None, answer_part.strip()

    def generate(self, messages: list[dict], thinking: Optional[bool] = None) -> LLMGenerateResult:
        use_thinking = self.thinking if thinking is None else thinking
        if isinstance(self.llm, OpenAI):
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    request_kwargs = {
                        "messages": messages,
                        "temperature": 0,
                        "model": self.model,
                    }
                    if use_thinking:
                        request_kwargs["extra_body"] = {"enable_thinking": True}
                    response = self.llm.chat.completions.create(**request_kwargs)
                    break
                except Exception as e:
                    logger.error(f"Attempt {attempt + 1} failed: {e}")
                    if attempt == max_retries - 1:
                        raise
                    sleep(3)
            message = response.choices[0].message
            content = self._normalize_text(message.content)
            reasoning_content = self._normalize_text(getattr(message, "reasoning_content", None)) or None
            if reasoning_content is None:
                parsed_reasoning, parsed_content = self._split_think_tag(content)
                reasoning_content = parsed_reasoning
                content = parsed_content
            return LLMGenerateResult(content=content, reasoning_content=reasoning_content)
        else:
            response = self.llm.create_chat_completion(messages=messages,temperature=0)
            content = self._normalize_text(response["choices"][0]["message"]["content"])
            reasoning_content, content = self._split_think_tag(content)
            return LLMGenerateResult(content=content, reasoning_content=reasoning_content)

def set_global_llm(api_key: str = None, base_url: str = None, model: str = None, lang: str = "English", thinking: bool = False):
    global GLOBAL_LLM
    GLOBAL_LLM = LLM(api_key=api_key, base_url=base_url, model=model, lang=lang, thinking=thinking)

def get_llm() -> LLM:
    if GLOBAL_LLM is None:
        logger.info("No global LLM found, creating a default one. Use `set_global_llm` to set a custom one.")
        set_global_llm()
    return GLOBAL_LLM