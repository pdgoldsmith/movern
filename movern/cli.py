"""Movern command-line interface."""
import sys


def main():
    """Main CLI entry point."""
    if len(sys.argv) < 2:
        print("Usage: movern <command>")
        print("Commands:")
        print("  ui    Launch the Streamlit assessment UI")
        sys.exit(1)

    command = sys.argv[1]

    if command == "ui":
        import subprocess

        import movern.ui.app as app_module

        subprocess.run(
            [
                sys.executable,
                "-m",
                "streamlit",
                "run",
                app_module.__file__,
                "--",
            ]
            + sys.argv[2:],
            check=True,
        )
    else:
        print(f"Unknown command: {command}")
        sys.exit(1)
