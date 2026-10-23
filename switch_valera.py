#!/usr/bin/env python3
import sys
import argparse
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
READY_DIR = HERE / "ready_voicepacks"
VENV_PYTHON = HERE / ".venv/bin/python"

PACKS = {
    "family": READY_DIR / "valera_family_pack.zip",
    "butler": READY_DIR / "valera_butler.zip",
    "gamer": READY_DIR / "valera_gamer.zip",
    "backup": READY_DIR / "backup_first_pass_custom_voicepack.zip",
    "original": READY_DIR / "ru_robot_new.zip",
}

def main():
    parser = argparse.ArgumentParser(description="Switch Valera voicepack on Xiaomi Robot Vacuum X20 Max")
    parser.add_argument("--list", action="store_true", help="List available voicepacks")
    parser.add_argument("--set", choices=list(PACKS.keys()), help="Set active voicepack")
    parser.add_argument("--country", default="de")
    parser.add_argument("--did", default="1173090359")
    args = parser.parse_args()

    if args.list or not args.set:
        print("\nAvailable voicepacks:")
        for key, path in PACKS.items():
            status = "READY" if path.exists() else "NOT BUILT YET"
            print(f"  • {key:10}: {path.name} [{status}]")
        print("\nUsage example: python switch_valera.py --set butler\n")
        return

    target = PACKS[args.set]
    if not target.exists():
        print(f"Error: {target} does not exist. Build it first!")
        sys.exit(1)

    print(f"\n[>>>] Switching Xiaomi X20 Max voicepack to: {args.set} ({target.name})...\n")
    cmd = [
        str(VENV_PYTHON),
        str(HERE / "voicepack_cycle.py"),
        "deploy",
        "--archive", str(target),
        "--country", args.country,
        "--did", args.did
    ]
    subprocess.run(cmd, check=True)

if __name__ == "__main__":
    main()
