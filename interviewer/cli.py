"""Run an interview in the terminal.

    python -m interviewer.cli
    python -m interviewer.cli                     # arm assigned at random
    python -m interviewer.cli --bridge | --no-bridge
    LLM_BACKEND=mock python -m interviewer.cli      # offline, no model needed
"""
import argparse

from .engine import InterviewSession


def main() -> None:
    ap = argparse.ArgumentParser(description="AI Conversational Interview on democracy")
    arm = ap.add_mutually_exclusive_group()
    arm.add_argument("--bridge", action="store_true", help="force the bridge arm")
    arm.add_argument("--no-bridge", action="store_true", help="force the control arm (no bridge question)")
    ap.add_argument("--context", help="country context pack, e.g. bangladesh, india, germany")
    args = ap.parse_args()

    s = InterviewSession(bridge=True if args.bridge else False if args.no_bridge else None, context=args.context)
    print(f"[bridge arm: {s.bridge}]")
    print(f"\nAI: {s.start()}\n")
    while not s.finished:
        try:
            answer = input("You: ")
        except (EOFError, KeyboardInterrupt):
            answer = "stop"
        print(f"\nAI: {s.respond(answer)}\n")
    print(f"[transcript saved to {s.save()}]")


if __name__ == "__main__":
    main()
