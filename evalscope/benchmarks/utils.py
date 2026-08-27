"""
Common utilities for benchmark adapters
"""

import re


def strip_thinking_blocks(text: str) -> str:
    """
    Strip Qwen3 thinking blocks from response text.

    Qwen3-Next models use <think>...</think> blocks for reasoning.
    This function removes these blocks to extract the actual response.

    Args:
        text: The raw response text from the model

    Returns:
        The text with thinking blocks removed
    """
    if not text:
        return text

    # Remove <think>...</think> blocks (including multiline)
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)

    # Also remove <thinking>...</thinking> blocks (alternative format)
    text = re.sub(r'<thinking>.*?</thinking>', '', text, flags=re.DOTALL)

    # Strip leading/trailing whitespace
    return text.strip()


def extract_code_block(text: str, language: str = 'python') -> str:
    """
    Extract code block from markdown-formatted text.
    If multiple code blocks exist, returns the longest one.

    Args:
        text: The text containing code blocks
        language: The programming language to look for (default: 'python')

    Returns:
        The extracted code, or the original text if no code block found
    """
    if not text:
        return text

    # First strip any thinking blocks
    text = strip_thinking_blocks(text)

    # Collect all code blocks
    all_code_blocks = []

    # Try to extract all language-specific code blocks
    pattern = f'```{language}\\s*\\n(.*?)```'
    matches = re.findall(pattern, text, re.DOTALL)
    for match in matches:
        if match.strip():
            all_code_blocks.append(match.strip())

    # Also try case-insensitive variant (e.g., Python vs python)
    if language.lower() != language:
        pattern = f'```{language.lower()}\\s*\\n(.*?)```'
        matches = re.findall(pattern, text, re.DOTALL)
        for match in matches:
            if match.strip() and match.strip() not in all_code_blocks:
                all_code_blocks.append(match.strip())

    # Try generic code blocks if no language-specific ones found
    if not all_code_blocks:
        pattern = '```\\s*\\n(.*?)```'
        matches = re.findall(pattern, text, re.DOTALL)
        for match in matches:
            if match.strip():
                all_code_blocks.append(match.strip())

    # Return the longest code block
    # This is typically the actual implementation rather than pseudocode
    if all_code_blocks:
        return max(all_code_blocks, key=len)

    # Return original text if no code block found
    return text


def normalize_answer(text: str) -> str:
    """
    Normalize answer text by stripping thinking blocks and extra whitespace.

    Args:
        text: The raw answer text

    Returns:
        Normalized answer text
    """
    if not text:
        return text

    # Strip thinking blocks
    text = strip_thinking_blocks(text)

    # Normalize whitespace
    text = ' '.join(text.split())

    return text.strip()