"""Execute every notebook cell, retaining genuine outputs for GitHub."""
from pathlib import Path
import nbformat
from nbclient import NotebookClient

root = Path(__file__).resolve().parent
path = root / "building_heating_load.ipynb"
notebook = nbformat.read(path, as_version=4)
NotebookClient(notebook, timeout=1800, kernel_name="python3",
               resources={"metadata": {"path": str(root)}}).execute()
nbformat.write(notebook, path)
print("Executed notebook saved:", path.name)
