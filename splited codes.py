"""
generate_steps.py
------------------
Splits one or more Jupyter notebooks (.ipynb) into individual, numbered
Python step-files (one file per notebook cell / pipeline step), and packages
them into a single folder that is ready to push to GitHub.

It also generates `app.py`, a simple point-and-click Tkinter GUI that lists
every step as a checkbox. You tick the step(s) you want, click "Run
Selected", and it executes them in order inside one shared Python session
(so variables created in an earlier step - X_train, model, etc. - are still
available to a later step, exactly like running cells in a notebook).

USAGE
-----
    python generate_steps.py notebook1.ipynb [notebook2.ipynb ...] -o output_folder

If no notebooks are given, every *.ipynb file in the current directory is used.
"""

import argparse
import json
import re
import shutil
from pathlib import Path


def slugify(text: str, max_len: int = 45) -> str:
    text = text.strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return (text or "step")[:max_len]


def cell_title(source_lines, index):
    """Pull a human-readable title from the first '#' comment line of a cell."""
    for line in source_lines:
        stripped = line.strip()
        if stripped.startswith("#"):
            title = stripped.lstrip("#").strip()
            if title:
                return title
    return f"Step {index}"


def split_notebook(nb_path: Path, dataset_tag: str, steps_dir: Path, manifest: list):
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    counter_start = len(manifest) + 1

    for i, cell in enumerate(nb.get("cells", [])):
        if cell.get("cell_type") != "code":
            continue

        source_lines = cell.get("source", [])
        source = "".join(source_lines).strip()

        if not source:
            continue  # skip empty placeholder cells

        title = cell_title(source_lines, i)
        step_num = len(manifest) + 1
        if title.lower().startswith("step "):
            fname = f"step_{step_num:02d}.py"
        else:
            fname = f"step_{step_num:02d}_{slugify(title)}.py"

        header = (
            f'"""\n'
            f"Step {step_num}: {title}\n"
            f"Source notebook : {nb_path.name}\n"
            f"Original cell   : #{i}\n"
            f'"""\n\n'
            f"from pipeline_shared import *  # noqa: F401,F403 "
            f"(shared imports / variables from earlier steps)\n\n"
        )

        (steps_dir / fname).write_text(header + source + "\n", encoding="utf-8")

        manifest.append(
            {
                "file": fname,
                "title": title,
                "notebook": nb_path.name,
                "dataset": dataset_tag,
            }
        )


def write_pipeline_shared(out_dir: Path):
    (out_dir / "pipeline_shared.py").write_text(
        '"""\n'
        "Common imports shared by every step file.\n"
        "Each step does `from pipeline_shared import *` so it has access to\n"
        "numpy / pandas / sklearn / matplotlib without repeating the imports,\n"
        "and so that `display()` works outside of Jupyter.\n"
        '"""\n\n'
        "import warnings\n"
        'warnings.filterwarnings("ignore")\n\n'
        "import numpy as np\n"
        "import pandas as pd\n"
        "import matplotlib.pyplot as plt\n"
        "import seaborn as sns\n"
        "import joblib\n\n"
        "from sklearn.datasets import load_breast_cancer, fetch_openml\n"
        "from sklearn.model_selection import (\n"
        "    train_test_split, StratifiedKFold, GridSearchCV,\n"
        "    cross_validate, cross_val_predict,\n"
        ")\n"
        "from sklearn.pipeline import Pipeline\n"
        "from sklearn.compose import ColumnTransformer, make_column_selector\n"
        "from sklearn.impute import SimpleImputer\n"
        "from sklearn.preprocessing import OneHotEncoder, StandardScaler\n"
        "from sklearn.dummy import DummyClassifier\n"
        "from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text\n"
        "from sklearn.ensemble import RandomForestClassifier\n"
        "from sklearn.inspection import permutation_importance\n"
        "from sklearn.metrics import (\n"
        "    confusion_matrix, ConfusionMatrixDisplay, accuracy_score,\n"
        "    balanced_accuracy_score, precision_score, recall_score, f1_score,\n"
        "    roc_auc_score, average_precision_score, RocCurveDisplay,\n"
        "    PrecisionRecallDisplay,\n"
        ")\n\n"
        "RANDOM_STATE = 42\n\n"
        "def display(x):\n"
        '    """Fallback for notebook-only display() when run as a plain script."""\n'
        "    print(x)\n",
        encoding="utf-8",
    )


def write_app(out_dir: Path, manifest: list):
    steps_literal = json.dumps(manifest, indent=4)
    app_code = f'''"""
app.py
------
Click-to-run GUI for the split pipeline steps.

Tick the checkbox next to any step (or several), click "Run Selected", and
each ticked step runs in order inside ONE shared Python session - so a step
that needs `X_train` from an earlier step will find it, just like re-running
cells in a notebook. Output (prints, errors) shows in the console pane below.
Plots open in normal matplotlib windows.

Run it with:   python app.py
"""

import sys
import traceback
import tkinter as tk
from tkinter import ttk, scrolledtext
from pathlib import Path

STEPS_DIR = Path(__file__).parent / "steps"
sys.path.insert(0, str(STEPS_DIR))  # so `from pipeline_shared import *` resolves

STEPS = {steps_literal}


class RedirectText:
    """Sends print() output into the Tkinter console widget."""
    def __init__(self, widget):
        self.widget = widget

    def write(self, text):
        self.widget.insert(tk.END, text)
        self.widget.see(tk.END)

    def flush(self):
        pass


class PipelineApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Decision Tree Lab - Step Runner")
        self.root.geometry("880x650")

        self.shared_namespace = {{"__name__": "__pipeline__"}}
        self.check_vars = []

        main = ttk.Frame(root, padding=10)
        main.pack(fill=tk.BOTH, expand=True)

        ttk.Label(
            main,
            text="Tick the step(s) you want to run, then click Run Selected.\\n"
                 "Steps run top-to-bottom in one shared session (like a notebook).",
            justify=tk.LEFT,
        ).pack(anchor="w", pady=(0, 8))

        list_frame = ttk.Frame(main)
        list_frame.pack(fill=tk.BOTH, expand=True)

        canvas = tk.Canvas(list_frame, borderwidth=0, height=300)
        scrollbar = ttk.Scrollbar(list_frame, orient="vertical", command=canvas.yview)
        scroll_frame = ttk.Frame(canvas)

        scroll_frame.bind(
            "<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        canvas.create_window((0, 0), window=scroll_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        last_dataset = None
        for step in STEPS:
            if step["dataset"] != last_dataset:
                ttk.Label(
                    scroll_frame,
                    text=f"\\n{{step['dataset']}}",
                    font=("TkDefaultFont", 9, "bold"),
                ).pack(anchor="w")
                last_dataset = step["dataset"]

            var = tk.BooleanVar()
            cb = ttk.Checkbutton(
                scroll_frame,
                text=f"{{step['file']}}  —  {{step['title']}}",
                variable=var,
            )
            cb.pack(anchor="w")
            self.check_vars.append((var, step))

        btn_frame = ttk.Frame(main)
        btn_frame.pack(fill=tk.X, pady=8)

        ttk.Button(btn_frame, text="Select All", command=self.select_all).pack(side=tk.LEFT)
        ttk.Button(btn_frame, text="Select None", command=self.select_none).pack(side=tk.LEFT, padx=6)
        ttk.Button(btn_frame, text="Run Selected", command=self.run_selected).pack(side=tk.LEFT, padx=6)
        ttk.Button(btn_frame, text="Run All", command=self.run_all).pack(side=tk.LEFT)
        ttk.Button(btn_frame, text="Reset Session", command=self.reset_session).pack(side=tk.RIGHT)

        ttk.Label(main, text="Console:").pack(anchor="w")
        self.console = scrolledtext.ScrolledText(main, height=16, bg="black", fg="lightgreen")
        self.console.pack(fill=tk.BOTH, expand=True)

    def select_all(self):
        for var, _ in self.check_vars:
            var.set(True)

    def select_none(self):
        for var, _ in self.check_vars:
            var.set(False)

    def reset_session(self):
        self.shared_namespace = {{"__name__": "__pipeline__"}}
        self.console.insert(tk.END, "\\n[Session reset - shared variables cleared]\\n")
        self.console.see(tk.END)

    def run_all(self):
        for var, _ in self.check_vars:
            var.set(True)
        self.run_selected()

    def run_selected(self):
        old_stdout, old_stderr = sys.stdout, sys.stderr
        sys.stdout = sys.stderr = RedirectText(self.console)
        try:
            for var, step in self.check_vars:
                if not var.get():
                    continue
                path = STEPS_DIR / step["file"]
                print(f"\\n=== Running {{step['file']}} ({{step['title']}}) ===\\n")
                try:
                    code = path.read_text(encoding="utf-8")
                    exec(compile(code, str(path), "exec"), self.shared_namespace)
                except Exception:
                    traceback.print_exc()
                    print(f"\\n--- Stopped: {{step['file']}} raised an error ---")
                    break
        finally:
            sys.stdout, sys.stderr = old_stdout, old_stderr


if __name__ == "__main__":
    root = tk.Tk()
    app = PipelineApp(root)
    root.mainloop()
'''
    (out_dir / "app.py").write_text(app_code, encoding="utf-8")


def write_readme(out_dir: Path, manifest: list):
    lines = [
        "# Decision Tree Lab — Step-by-Step Pipeline",
        "",
        "This folder was generated automatically by splitting the original Jupyter",
        "notebook(s) into one Python file per step (per cell). Each file lives in",
        "`steps/` and is numbered in the original notebook order.",
        "",
        "## How to run",
        "",
        "### Option A — Click-to-run GUI (recommended)",
        "```bash",
        "pip install -r requirements.txt",
        "python app.py",
        "```",
        "A window opens listing every step as a checkbox. Tick one, several, or",
        "all of them, then click **Run Selected**. Steps execute top-to-bottom in a",
        "single shared session, so a later step can use variables created by an",
        "earlier one (just like re-running cells in a notebook).",
        "",
        "### Option B — Run a single step from the command line",
        "```bash",
        "cd steps",
        "python step_05_train_test_split.py   # example",
        "```",
        "Note: most steps depend on variables created by earlier steps (`X`, `y`,",
        "`X_train`, models, etc.), so for command-line use, run the steps for a",
        "given dataset in numeric order, in the same Python session (e.g. `python",
        "-i` and repeated `exec(open(...).read())`, or just use `app.py`).",
        "",
        "## Notes",
        "- Steps for the **UCI Heart Disease** dataset call `fetch_openml(...)`",
        "  and therefore require an internet connection.",
        "- `pipeline_shared.py` holds the imports/constants common to every step.",
        "",
        "## Step index",
        "",
        "| # | File | Step | Dataset |",
        "|---|------|------|---------|",
    ]
    for i, step in enumerate(manifest, start=1):
        lines.append(f"| {i} | `steps/{step['file']}` | {step['title']} | {step['dataset']} |")

    (out_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_requirements(out_dir: Path):
    (out_dir / "requirements.txt").write_text(
        "numpy\npandas\nmatplotlib\nseaborn\nscikit-learn\njoblib\n",
        encoding="utf-8",
    )


def write_gitignore(out_dir: Path):
    (out_dir / ".gitignore").write_text(
        "__pycache__/\n*.pyc\n.ipynb_checkpoints/\n.DS_Store\n",
        encoding="utf-8",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("notebooks", nargs="*", help="Path(s) to .ipynb file(s)")
    parser.add_argument("-o", "--output", default="pipeline_steps", help="Output folder")
    args = parser.parse_args()

    notebooks = [Path(p) for p in args.notebooks] or sorted(Path(".").glob("*.ipynb"))
    if not notebooks:
        raise SystemExit("No .ipynb files found. Pass paths explicitly.")

    out_dir = Path(args.output)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    steps_dir = out_dir / "steps"
    steps_dir.mkdir(parents=True)

    manifest = []
    for nb_path in notebooks:
        tag = nb_path.stem
        split_notebook(nb_path, tag, steps_dir, manifest)

    write_pipeline_shared(steps_dir)
    write_app(out_dir, manifest)
    write_readme(out_dir, manifest)
    write_requirements(out_dir)
    write_gitignore(out_dir)

    print(f"Done. {len(manifest)} step files written to: {out_dir}/steps/")
    print(f"GUI runner: {out_dir}/app.py")


if __name__ == "__main__":
    main()
