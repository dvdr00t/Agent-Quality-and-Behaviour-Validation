"""CLI application for the retail support system."""

from __future__ import annotations

import argparse
import logging

from retail_support.config import DEFAULT_MODEL, SupportSettings
from retail_support.runtime import RetailSupportOrchestrator


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Retail support multi-agent system")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"OpenAI model name. Default: {DEFAULT_MODEL}")
    parser.add_argument(
        "--agent",
        default="supervisor",
        choices=["supervisor", "knowledge", "orders", "safety"],
        help="Agent target for direct interaction.",
    )
    parser.add_argument("--message", help="Send a single message and exit.")
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Start an interactive terminal session.",
    )
    return parser


def build_orchestrator(model_name: str) -> RetailSupportOrchestrator:
    settings = SupportSettings.from_env(model_name_override=model_name)
    return RetailSupportOrchestrator(settings=settings)


def run_interactive(orchestrator: RetailSupportOrchestrator, initial_target: str) -> None:
    target = initial_target
    session = orchestrator.start_session()
    print("Retail support console ready.")
    print("Use /agent supervisor|knowledge|orders|safety to change the target agent.")
    print("Use /exit to close the session.")

    while True:
        user_message = input("\nYou > ").strip()
        if user_message.lower() in {"/exit", "exit", "quit"}:
            print("Session closed.")
            return

        if user_message.startswith("/agent "):
            target = user_message.split(" ", 1)[1].strip().lower()
            print(f"Target agent set to: {target}")
            continue

        reply = orchestrator.reply(session=session, user_message=user_message, target=target)
        print(f"\n{reply.handled_by} > {reply.text}")
        if reply.route and target == "supervisor":
            print(f"Route: {', '.join(reply.route)}")
        if reply.tool_calls:
            print(f"Tools: {', '.join(reply.tool_calls)}")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")
    parser = build_parser()
    args = parser.parse_args()
    orchestrator = build_orchestrator(model_name=args.model)

    if args.message:
        session = orchestrator.start_session()
        reply = orchestrator.reply(session=session, user_message=args.message, target=args.agent)
        print(reply.text)
        return

    run_interactive(orchestrator=orchestrator, initial_target=args.agent)


if __name__ == "__main__":
    main()
