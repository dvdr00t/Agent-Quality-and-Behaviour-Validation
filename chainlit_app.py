"""Chainlit user interface for the retail support system."""

from __future__ import annotations

import chainlit as cl

from retail_support.config import SupportSettings
from retail_support.runtime import RetailSupportOrchestrator


AVAILABLE_TARGETS = {"supervisor", "knowledge", "orders", "safety"}


@cl.on_chat_start
async def on_chat_start() -> None:
    try:
        settings = SupportSettings.from_env()
        orchestrator = RetailSupportOrchestrator(settings=settings)
        session = orchestrator.start_session()
    except Exception as exc:
        await cl.Message(content=f"Startup failed: {exc}").send()
        return

    cl.user_session.set("orchestrator", orchestrator)
    cl.user_session.set("support_session", session)
    cl.user_session.set("target", "supervisor")

    await cl.Message(
        author="System",
        content=(
            "Retail support workspace connected.\n\n"
            "Use `/agent supervisor`, `/agent knowledge`, `/agent orders`, or `/agent safety` "
            "to choose which subsystem you want to talk to."
        ),
    ).send()


@cl.on_message
async def on_message(message: cl.Message) -> None:
    orchestrator = cl.user_session.get("orchestrator")
    support_session = cl.user_session.get("support_session")
    target = cl.user_session.get("target", "supervisor")

    if orchestrator is None or support_session is None:
        await cl.Message(author="System", content="Session not initialized. Start a new chat.").send()
        return

    content = message.content.strip()
    if content.startswith("/agent "):
        requested_target = content.split(" ", 1)[1].strip().lower()
        if requested_target not in AVAILABLE_TARGETS:
            await cl.Message(
                author="System",
                content="Unsupported agent. Valid options: supervisor, knowledge, orders, safety.",
            ).send()
            return
        cl.user_session.set("target", requested_target)
        await cl.Message(author="System", content=f"Active target set to `{requested_target}`.").send()
        return

    reply = await cl.make_async(orchestrator.reply)(
        session=support_session,
        user_message=content,
        target=target,
    )

    footer = ""
    if target == "supervisor" and reply.route:
        footer = f"\n\n---\nRoute: {', '.join(reply.route)}"

    await cl.Message(author=reply.handled_by, content=f"{reply.text}{footer}").send()
