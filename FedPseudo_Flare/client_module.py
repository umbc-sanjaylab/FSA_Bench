import sys
import runpy
from pathlib import Path


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent

    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    runpy.run_module(
        "FedPseudo_Flare.client.client",
        run_name="__main__"
    )