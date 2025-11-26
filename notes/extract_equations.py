#!/usr/bin/env python3
import re
import sys
import shutil
import tempfile
import subprocess
import argparse
from pathlib import Path

OUTPUT_DIR = Path("./equations")

# Rendering knobs for the standalone PDFs.
# SIZE_CMD bumps the baseline text size; WIDTH/HEIGHT make sure TeX never clips
# wide expressions even after upscaling; BORDER pads the crop box so Keynote
# doesn't trim glyphs that reach the edge.
SIZE_CMD = r"\fontsize{36pt}{44pt}\selectfont"
PAPER_WIDTH = "60in"
PAPER_HEIGHT = "40in"
BORDER_PT = 18
NUMBERED_ENVIRONMENTS = (
    "equation",
    "align",
    "alignat",
    "gather",
    "multline",
    "flalign",
)

PREAMBLE_TEMPLATE = r"""
\special{papersize={__WIDTH__,__HEIGHT__}}
\setlength{\paperwidth}{__WIDTH__}
\setlength{\paperheight}{__HEIGHT__}
\setlength{\textwidth}{\paperwidth}
\setlength{\linewidth}{\paperwidth}
\setlength{\columnwidth}{\paperwidth}
\setlength{\textheight}{\paperheight}
\setlength{\pdfpagewidth}{\paperwidth}
\setlength{\pdfpageheight}{\paperheight}
\usepackage{amsmath,amssymb,amsfonts,mathtools}
\usepackage{xcolor}
\usepackage{pagecolor}
\usepackage{anyfontsize}
"""

def get_preamble(transparent=False):
    base = PREAMBLE_TEMPLATE.replace("__WIDTH__", PAPER_WIDTH).replace("__HEIGHT__", PAPER_HEIGHT)
    if transparent:
        # White text, transparent background (default pagecolor is none/transparent)
        # We use \nopagecolor to ensure no background color is painted if pagecolor package set one.
        colors = r"\nopagecolor\color{white}"
    else:
        # White text, black background
        colors = r"\pagecolor{black}\color{white}"
    return base + "\n" + colors

def read_file(path: Path) -> str:
    return path.read_text(encoding="utf-8")

def find_equations(tex: str):
    r"""
    Grab display equations:
      - equation / equation*
      - align / align*
      - \[ ... \]
      - $$ ... $$
    """
    patterns = [
        r"\\begin\{equation\*?\}.*?\\end\{equation\*?\}",
        r"\\begin\{align\*?\}.*?\\end\{align\*?\}",
        r"\\\[.*?\\\]",
        r"\$\$.*?\$\$",
    ]
    matches = []
    for pat in patterns:
        for m in re.finditer(pat, tex, flags=re.DOTALL):
            matches.append((m.start(), m.group(0)))
    matches.sort(key=lambda x: x[0])
    return [m[1] for m in matches]

def disable_numbering(eq_src: str) -> str:
    """
    Force otherwise numbered math environments into their * (unnumbered) forms
    and strip explicit numbering commands like \\tag{} or \\label{}.
    """
    for env in NUMBERED_ENVIRONMENTS:
        eq_src = re.sub(
            rf"(\\begin\{{{env}\}})(?!\*)",
            rf"\\begin{{{env}*}}",
            eq_src,
        )
        eq_src = re.sub(
            rf"(\\end\{{{env}\}})(?!\*)",
            rf"\\end{{{env}*}}",
            eq_src,
        )

    # Remove tags/labels that would still reserve horizontal space.
    strip_patterns_multiline = [
        r"^\s*\\tag\*?\{.*?\}\s*$",
        r"^\s*\\label\{.*?\}\s*$",
        r"^\s*\\eqno\s*\{.*?\}\s*$",
    ]
    for pat in strip_patterns_multiline:
        eq_src = re.sub(pat, "", eq_src, flags=re.MULTILINE)

    strip_patterns_inline = [
        r"\\tag\*?\{.*?\}",
        r"\\label\{.*?\}",
        r"\\eqno\s*\{.*?\}",
    ]
    for pat in strip_patterns_inline:
        eq_src = re.sub(pat, "", eq_src, flags=re.DOTALL)

    # Collapse blank lines that can introduce spurious \par tokens in math mode.
    eq_src = re.sub(r"\n{2,}", "\n", eq_src)

    return eq_src

def build_tex_for_equation(eq_src: str, preamble: str) -> str:
    # 12pt base font, varwidth shrinks page to natural width of the content.
    return rf"""\documentclass[12pt,varwidth,border={BORDER_PT}pt]{{standalone}}
{preamble}
\begin{{document}}
{SIZE_CMD}
{eq_src}
\end{{document}}
"""

def run_pdflatex(tex_source: str, jobname: str):
    """
    Compile LaTeX source via a temporary work directory so no .tex artifacts end
    up in OUTPUT_DIR. Only the final PDF is moved into place.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        tex_path = tmpdir_path / f"{jobname}.tex"
        tex_path.write_text(tex_source, encoding="utf-8")

        subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", tex_path.name],
            cwd=tmpdir_path,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )

        pdf_path = tmpdir_path / f"{jobname}.pdf"
        dest_pdf = OUTPUT_DIR / f"{jobname}.pdf"
        shutil.move(pdf_path, dest_pdf)

def main():
    parser = argparse.ArgumentParser(description="Extract LaTeX equations to PDF.")
    parser.add_argument("input_tex", type=Path, help="Input .tex file")
    parser.add_argument("--transparent", action="store_true", help="Output white-on-transparent PDF")
    args = parser.parse_args()

    if not args.input_tex.exists():
        print(f"Error: File {args.input_tex} not found.")
        sys.exit(1)

    tex_source = read_file(args.input_tex)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    equations = find_equations(tex_source)
    print(f"Found {len(equations)} equations.")

    preamble = get_preamble(transparent=args.transparent)

    total = len(equations)
    sys.stdout.write(f"\r[{'-'*40}] 0/{total}")
    sys.stdout.flush()

    for i, eq in enumerate(equations, start=1):
        idx = f"{i:03d}"
        cleaned_eq = disable_numbering(eq)
        tex_source_eq = build_tex_for_equation(cleaned_eq, preamble)

        jobname = f"eq_{idx}"
        try:
            run_pdflatex(tex_source_eq, jobname)
        except subprocess.CalledProcessError:
            sys.stdout.write("\n")
            print(f"Failed to compile eq_{idx}")
        
        bar_len = 40
        filled_len = int(bar_len * i // total)
        bar = '=' * filled_len + '-' * (bar_len - filled_len)
        sys.stdout.write(f"\r[{bar}] {i}/{total}")
        sys.stdout.flush()

    print()
    print("Done. PDFs are in ./equations")

if __name__ == "__main__":
    main()
