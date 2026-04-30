#!/usr/bin/env python3
"""
NeuralForge — unified entry point.

Dashboard mode (default):
    python main.py
    python main.py --port 8080 --open-browser

CLI pass-through (same as running `neural <cmd>`):
    python main.py train   --config runs/my_exp/config.yaml
    python main.py serve   --config runs/my_exp/config.yaml
    python main.py init    --model mlp --name my_exp
    python main.py list
    python main.py status  1
    python main.py export  --config ...
    python main.py evaluate --config ...
    python main.py dashboard --port 7860
"""

from __future__ import annotations

import sys


# CLI sub-commands that get passed straight through to the typer/click app
_CLI_COMMANDS = {
    "train", "serve", "evaluate", "init",
    "list", "status", "export", "dashboard",
    "--version", "--help", "-h",
}


def main() -> None:
    # If the first positional arg is a known CLI command, delegate to the CLI.
    if len(sys.argv) > 1 and sys.argv[1] in _CLI_COMMANDS:
        from neural_platform.cli.commands import cli
        cli(standalone_mode=True)
        return

    # Otherwise start the dashboard directly.
    import argparse
    import threading
    import webbrowser

    import uvicorn

    from neural_platform.web.app import create_dashboard_app

    parser = argparse.ArgumentParser(
        prog="neural",
        description="NeuralForge Dashboard — build, train and deploy neural networks",
    )
    parser.add_argument(
        "--port", "-p",
        default=7860, type=int,
        help="Dashboard port (default: 7860)",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Bind host (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--output-dir", "-o",
        default="runs",
        dest="output_dir",
        help="Experiments output directory (default: runs/)",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable hot-reload (development mode)",
    )
    parser.add_argument(
        "--open-browser",
        action="store_true",
        dest="open_browser",
        help="Automatically open a browser tab",
    )
    args = parser.parse_args()

    app = create_dashboard_app(args.output_dir)
    url = f"http://{args.host}:{args.port}"

    print()
    print("  ⚡  NeuralForge Dashboard")
    print(f"      → {url}")
    print()
    print("  Tip: start training directly from the dashboard,")
    print(f"  or run:  python main.py train --config <path>")
    print()

    if args.open_browser:
        def _open_later() -> None:
            import time
            time.sleep(1.4)          # wait for uvicorn to bind
            webbrowser.open(url)

        threading.Thread(target=_open_later, daemon=True).start()

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="warning",         # keep the console tidy
    )


if __name__ == "__main__":
    main()
