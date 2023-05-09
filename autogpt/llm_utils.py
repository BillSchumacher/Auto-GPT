from __future__ import annotations

from typing import List, Optional

import openai
import requests
from colorama import Fore
from tenacity import (
    retry,
    retry_if_not_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)

from autogpt.api_manager import ApiManager
from autogpt.config import Config
from autogpt.logs import logger
from autogpt.types.openai import Message


def call_ai_function(
    function: str, args: list, description: str, model: str | None = None
) -> str:
    """Call an AI function

    This is a magic function that can do anything with no-code. See
    https://github.com/Torantulino/AI-Functions for more info.

    Args:
        function (str): The function to call
        args (list): The arguments to pass to the function
        description (str): The description of the function
        model (str, optional): The model to use. Defaults to None.

    Returns:
        str: The response from the function
    """
    cfg = Config()
    if model is None:
        model = cfg.smart_llm_model
    # For each arg, if any are None, convert to "None":
    args = [str(arg) if arg is not None else "None" for arg in args]
    # parse args to comma separated string
    args: str = ", ".join(args)
    messages: list[Message] = [
        Message(
            "system",
            f"You are now the following python function: ```# {description}"
            f"\n{function}```\n\nOnly respond with your `return` value.",
        ),
        Message("user", args),
    ]

    return create_chat_completion(model=model, messages=messages, temperature=0)


def handle_chat_completion(cfg, messages, model, temperature, max_tokens) -> str | None:
    for plugin in cfg.plugins:
        if not plugin.can_handle_chat_completion(
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        ):
            continue
        message = plugin.handle_chat_completion(
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        if message is not None:
            return message


def create_chat_completion(
    messages: List[Message],  # type: ignore
    model: Optional[str] = None,
    temperature: float = None,
    max_tokens: Optional[int] = None,
    use_fastchat: bool = False,
) -> str:
    """Create a chat completion using the OpenAI API

    Args:
        messages (List[Message]): The messages to send to the chat completion
        model (str, optional): The model to use. Defaults to None.
        temperature (float, optional): The temperature to use. Defaults to 0.9.
        max_tokens (int, optional): The max tokens to use. Defaults to None.

    Returns:
        str: The response from the chat completion
    """
    cfg = Config()
    if temperature is None:
        temperature = cfg.temperature

    if cfg.debug_mode:
        print(
            f"{Fore.GREEN}Creating chat completion with model {model}, temperature {temperature}, max_tokens {max_tokens}{Fore.RESET}"
        )
    if message := handle_chat_completion(
        cfg, messages, model, temperature, max_tokens
    ):
        return message
    api_manager = ApiManager()
    response = None
    llm_messages = [message.to_dict() for message in messages]
    if cfg.use_azure:
        response = api_manager.create_chat_completion(
            deployment_id=cfg.get_azure_deployment_id_for_model(model),
            model=model,
            messages=llm_messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    elif use_fastchat:
        response = fastchat_chat_completion(
            model=cfg.fastchat_model,
            messages=llm_messages,
            temperature=temperature,
            max_tokens=max_tokens or cfg.fastchat_token_limit,
        )
    else:
        response = api_manager.create_chat_completion(
            model=model,
            messages=llm_messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    if response is None:
        logger.typewriter_log(
            "FAILED TO GET RESPONSE FROM OPENAI",
            Fore.RED,
            f"Auto-GPT has failed to get a response from OpenAI's services. Try running Auto-GPT again, and if the problem the persists try running it with `{Fore.CYAN}--debug{Fore.RESET}`.",
        )
        logger.double_check()
    if isinstance(response, dict):
        return response["choices"][0]["message"]["content"]
    resp = response.choices[0].message["content"]
    for plugin in cfg.plugins:
        if not plugin.can_handle_on_response():
            continue
        resp = plugin.on_response(resp)
    return resp


def get_ada_embedding(text: str) -> List[float]:
    """Get an embedding from the ada model.

    Args:
        text (str): The text to embed.

    Returns:
        List[float]: The embedding.
    """
    cfg = Config()
    model = "text-embedding-ada-002"
    text = text.replace("\n", " ")

    if cfg.use_azure:
        kwargs = {"engine": cfg.get_azure_deployment_id_for_model(model)}
    else:
        kwargs = {"model": model}

    embedding = create_embedding(text, **kwargs)
    api_manager = ApiManager()
    api_manager.update_cost(
        prompt_tokens=embedding.usage.prompt_tokens,
        completion_tokens=0,
        model=model,
    )
    return embedding["data"][0]["embedding"]


@retry(
    wait=wait_random_exponential(min=1, max=20),
    stop=stop_after_attempt(10),
    retry=retry_if_not_exception_type(openai.InvalidRequestError),
)
def create_embedding(
    text: str,
    *_,
    **kwargs,
) -> openai.Embedding:
    """Create an embedding using the OpenAI API

    Args:
        text (str): The text to embed.
        kwargs: Other arguments to pass to the OpenAI API embedding creation call.

    Returns:
        openai.Embedding: The embedding object.
    """
    cfg = Config()
    return openai.Embedding.create(
        input=[text],
        api_key=cfg.openai_api_key,
        **kwargs,
    )


@retry(
    wait=wait_random_exponential(min=1, max=20),
    stop=stop_after_attempt(10),
)
def fastchat_chat_completion(
    model,
    messages,
    temperature=0.8,
    max_tokens=4096,
):
    cfg = Config()
    resp = requests.post(
        f"http://{cfg.fastchat_host}:8000/v1/chat/completions",
        json={
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        },
    )
    json_resp = resp.json()
    logger.debug(f"{json_resp}")
    return json_resp
