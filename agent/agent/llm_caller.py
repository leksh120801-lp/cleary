import asyncio

from slack_sdk.web.chat_stream import ChatStream

from listeners.assistant.mcp_client import run_agent_turn


def call_llm(streamer: ChatStream, prompts: list[dict]):
    """Run an A11y Ally agent turn (Gemini reasoning + a11y-mcp tool calls) and stream the result."""
    asyncio.run(run_agent_turn(streamer, prompts))
