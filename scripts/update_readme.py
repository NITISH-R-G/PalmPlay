import json
import re
from pathlib import Path


def generate_readme_content(graph: dict, diagrams: dict) -> str:
    content = []

    # Header
    content.append("# 🤖 Automated Repository Overview\n")
    content.append(
        "*This README is automatically generated and updated by AI and CI/CD pipelines.*"
    )
    content.append("\n## 📊 System Architecture\n")
    content.append("### Static View\n")
    content.append(
        "```mermaid\n" + diagrams.get("architecture", "graph TD;\n  App") + "\n```\n"
    )
    content.append("### Interactive View\n")
    content.append(
        "An interactive, clickable node diagram has been generated. [View Interactive Diagram](./diagrams/interactive_architecture.html)\n"
    )

    # Technology Stack
    content.append("## 🛠️ Technology Stack\n")
    if "frameworks" in graph and graph["frameworks"]:
        for fw in set(graph["frameworks"]):
            content.append(f"- **{fw}**")
    else:
        content.append("- Python ecosystem")
    content.append("\n")

    # Environment Variables
    if "environment_variables" in graph and graph["environment_variables"]:
        content.append("## ⚙️ Environment Variables\n")
        content.append("| Variable | Description |")
        content.append("| -------- | ----------- |")
        for env in graph["environment_variables"]:
            content.append(f"| `{env}` | Configured externally |")
        content.append("\n")

    # Setup Instructions
    content.append("## 🚀 Quick Start\n")
    content.append(
        "1. **Install dependencies:**\n   ```bash\n   pip install -r requirements.txt\n   ```\n"
    )
    content.append(
        "2. **Set up environment:**\n   Copy `.env.example` or `.env.bak` to `.env` and configure.\n"
    )
    content.append(
        "3. **Run the application:**\n   ```bash\n   python app.py  # or streamlit run app.py depending on your framework\n   ```\n"
    )

    return "\n".join(content)


def main():
    graph_path = Path("knowledge_graph.json")
    if not graph_path.exists():
        print("Knowledge graph not found. Run analyze_repo.py first.")
        return

    with open(graph_path, "r", encoding="utf-8") as f:
        graph = json.load(f)

    diagrams = {}
    arch_path = Path("diagrams/architecture.mermaid")
    if arch_path.exists():
        with open(arch_path, "r", encoding="utf-8") as f:
            diagrams["architecture"] = f.read()

    new_content = generate_readme_content(graph, diagrams)

    readme_path = Path("README.md")
    if readme_path.exists():
        with open(readme_path, "r", encoding="utf-8") as f:
            existing = f.read()

        # Inject our automated content at the end or replace a section
        # For this setup, we'll append a dedicated Automated Section if it doesn't exist
        marker = "<!-- AUTO-GENERATED-DOCS-START -->"
        end_marker = "<!-- AUTO-GENERATED-DOCS-END -->"

        if marker in existing and end_marker in existing:
            # Replace existing section
            pattern = re.compile(f"{marker}.*?{end_marker}", re.DOTALL)
            replacement = f"{marker}\n{new_content}\n{end_marker}"
            updated = pattern.sub(replacement, existing)
        else:
            updated = existing + f"\n\n{marker}\n{new_content}\n{end_marker}\n"
    else:
        updated = f"<!-- AUTO-GENERATED-DOCS-START -->\n{new_content}\n<!-- AUTO-GENERATED-DOCS-END -->\n"

    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(updated)

    print("README.md updated successfully.")


if __name__ == "__main__":
    main()
