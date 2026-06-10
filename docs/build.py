"""Build static HTML documentation from Markdown files.

Usage: python docs/build.py

Reads all .md files from docs/ (excluding _config.yml, _layouts/, build.py),
converts them to HTML with a shared template, and writes to _site/.
"""

import os
import re
import glob
import markdown


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{title} | AXCGH-Engine</title>
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/github-markdown-css@5/github-markdown.min.css">
  <style>
    body {{ max-width: 980px; margin: 0 auto; padding: 45px; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif; }}
    .nav {{ margin-bottom: 2em; padding-bottom: 1em; border-bottom: 1px solid #eaecef; }}
    .nav a {{ margin-right: 1.2em; text-decoration: none; color: #0366d6; font-weight: 500; }}
    .nav a:hover {{ text-decoration: underline; }}
    .nav .active {{ color: #24292e; font-weight: 700; }}
    footer {{ margin-top: 3em; padding-top: 1em; border-top: 1px solid #eaecef; color: #586069; font-size: 0.85em; }}
  </style>
</head>
<body>
  <nav class="nav">
    <a href="index.html">Home</a>
    <a href="getting_started.html">Getting Started</a>
    <a href="api_reference.html">API Reference</a>
    <a href="architecture.html">Architecture</a>
    <a href="https://github.com/Lishoulan/AXCGH-Engine">GitHub</a>
  </nav>
  <article class="markdown-body">
    {content}
  </article>
  <footer>
    &copy; AXCGH-Engine &mdash; Deep Learning Computer-Generated Holography Engine
  </footer>
</body>
</html>"""


def strip_front_matter(text: str) -> tuple:
    """Remove YAML front matter and return (metadata_dict, remaining_text)."""
    meta = {}
    if text.startswith("---"):
        parts = text.split("---", 2)
        if len(parts) >= 3:
            for line in parts[1].strip().splitlines():
                if ":" in line:
                    key, val = line.split(":", 1)
                    meta[key.strip()] = val.strip()
            return meta, parts[2].strip()
    return meta, text


def extract_title(text: str, fallback: str) -> str:
    """Extract first # heading from markdown text."""
    for line in text.splitlines():
        if line.startswith("# "):
            return line[2:].strip()
    return fallback


def build_docs():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(script_dir, "..", "_site")
    os.makedirs(out_dir, exist_ok=True)

    # Collect markdown files
    md_files = []
    for f in sorted(glob.glob(os.path.join(script_dir, "*.md"))):
        basename = os.path.basename(f)
        if basename.startswith("_"):
            continue
        md_files.append((f, basename))

    # Also convert README.md if it exists at root
    readme_path = os.path.join(script_dir, "..", "README.md")
    if os.path.exists(readme_path):
        md_files.append((readme_path, "README.md"))

    md = markdown.Markdown(extensions=["fenced_code", "tables", "toc", "codehilite"],
                           extension_configs={"codehilite": {"css_class": "highlight"}})

    for filepath, filename in md_files:
        with open(filepath, "r", encoding="utf-8") as f:
            raw = f.read()

        meta, body = strip_front_matter(raw)
        title = meta.get("title", extract_title(body, "AXCGH-Engine"))

        # Convert markdown to HTML
        md.reset()
        html_body = md.convert(body)

        # Output filename
        out_name = filename.replace(".md", ".html")
        if out_name == "README.html":
            # Don't overwrite index; skip or link from index
            continue

        html = HTML_TEMPLATE.format(title=title, content=html_body)

        out_path = os.path.join(out_dir, out_name)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(html)
        print(f"  Built: {out_name}")

    # Build index page
    index_html = HTML_TEMPLATE.format(
        title="AXCGH-Engine",
        content="""
<h1>AXCGH-Engine</h1>
<p>Deep Learning Computer-Generated Holography Engine</p>
<h2>Documentation</h2>
<ul>
  <li><a href="getting_started.html">Getting Started</a> — Installation and first hologram</li>
  <li><a href="api_reference.html">API Reference</a> — Full API documentation</li>
  <li><a href="architecture.html">Architecture</a> — Architecture deep dive</li>
</ul>
<h2>Links</h2>
<ul>
  <li><a href="https://github.com/Lishoulan/AXCGH-Engine">GitHub Repository</a></li>
</ul>
"""
    )
    with open(os.path.join(out_dir, "index.html"), "w", encoding="utf-8") as f:
        f.write(index_html)
    print("  Built: index.html")

    print(f"\nDocs built successfully in {out_dir}")


if __name__ == "__main__":
    build_docs()
