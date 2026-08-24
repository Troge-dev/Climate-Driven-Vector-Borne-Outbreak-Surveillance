import re
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent
md_files = list(repo_root.glob("*.md")) + list(repo_root.glob("docs/*.md"))
nb_files = list(repo_root.glob("notebooks/*.ipynb")) + list(repo_root.glob("*.ipynb"))

def slugify(text):
    text = text.strip().lower()
    text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)
    text = re.sub(r'[\`\*\_]', '', text)
    text = re.sub(r'\$[^\$]+\$', '', text)
    text = re.sub(r'[^\w\s-]', '', text)
    text = re.sub(r'\s+', '-', text)
    return text

emoji_pattern = re.compile(r'[\U00010000-\U0010ffff]|[\u2600-\u27bf]|[\u2300-\u23ff]|[\u2b50-\u2b55]')

print("=== CHECKING HEADINGS FOR DECORATIVE EMOJIS ===")
heading_count = 0
emoji_heading_count = 0

for f in md_files:
    content = f.read_text(encoding="utf-8")
    for line_num, line in enumerate(content.splitlines(), 1):
        if line.startswith("#"):
            heading_count += 1
            emojis = emoji_pattern.findall(line)
            if emojis:
                emoji_heading_count += 1
                print(f"EMOJI IN HEADING: {f.name}:{line_num} -> {line}")

import json
for nb in nb_files:
    data = json.loads(nb.read_text(encoding="utf-8"))
    for c_idx, cell in enumerate(data["cells"]):
        if cell["cell_type"] == "markdown":
            for line in "".join(cell["source"]).splitlines():
                if line.startswith("#"):
                    heading_count += 1
                    emojis = emoji_pattern.findall(line)
                    if emojis:
                        emoji_heading_count += 1
                        print(f"EMOJI IN NOTEBOOK HEADING: {nb.name}:cell_{c_idx} -> {line}")

print(f"Checked {heading_count} total headings across {len(md_files)} markdown files. Found {emoji_heading_count} headings with emojis.")

print("\n=== CHECKING ALL INTERNAL AND RELATIVE LINKS ===")
link_issues = 0
for f in md_files:
    content = f.read_text(encoding="utf-8")
    headings = [line.lstrip("#").strip() for line in content.splitlines() if line.startswith("#")]
    slugs = {slugify(h) for h in headings}
    
    # Internal anchors: [text](#anchor)
    anchors = re.findall(r'\[([^\]]+)\]\(#([^\)]+)\)', content)
    for txt, anc in anchors:
        if anc not in slugs:
            print(f"ANCHOR MISMATCH in {f.name}: ['{txt}'](#{anc})")
            link_issues += 1
        else:
            print(f"VALID ANCHOR in {f.name}: ['{txt}'](#{anc})")
            
    # Relative file links: [text](path/file.md)
    file_links = re.findall(r'\[([^\]]+)\]\(([^#\):]+)(?:#([^\)]+))?\)', content)
    for txt, target_path_str, anc in file_links:
        if target_path_str.startswith("http") or target_path_str.startswith("mailto"):
            continue
        target_path = (f.parent / target_path_str).resolve()
        if not target_path.exists():
            print(f"BROKEN FILE LINK in {f.name}: ['{txt}']({target_path_str}) -> {target_path} not found")
            link_issues += 1
        elif anc:
            target_content = target_path.read_text(encoding="utf-8")
            target_headings = [line.lstrip("#").strip() for line in target_content.splitlines() if line.startswith("#")]
            target_slugs = {slugify(h) for h in target_headings}
            if anc not in target_slugs:
                print(f"BROKEN ANCHOR LINK in {f.name}: ['{txt}']({target_path_str}#{anc}) -> anchor not in target")
                link_issues += 1

if link_issues == 0:
    print("\n[ALL CHECKS PASSED] All links and anchors are valid and resolved!")
else:
    print(f"\n[WARNING] Found {link_issues} link issues.")
