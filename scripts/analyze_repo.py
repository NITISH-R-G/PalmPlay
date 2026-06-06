import os
import json
import ast
from pathlib import Path
from typing import Dict, List, Set, Any


class RepoAnalyzer:
    def __init__(self, root_dir: str):
        self.root_dir = Path(root_dir)
        self.knowledge_graph = {
            "modules": {},
            "frameworks": [],
            "apis": [],
            "databases": [],
            "deployment": [],
        }
        self.exclude_dirs = {
            ".git",
            "__pycache__",
            "venv",
            "env",
            "node_modules",
            ".github",
            "static",
            "local_music",
            "scripts",
        }

    def is_python_file(self, filepath: Path) -> bool:
        return filepath.suffix == ".py"

    def analyze(self):
        self._scan_dependencies()
        self._scan_python_files()
        self._detect_environment()
        self._save_graph()

    def _scan_dependencies(self):
        req_file = self.root_dir / "requirements.txt"
        if req_file.exists():
            with open(req_file, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
                # A simple check for frameworks based on common packages
                if "streamlit" in content.lower():
                    self.knowledge_graph["frameworks"].append("Streamlit")
                if "fastapi" in content.lower():
                    self.knowledge_graph["frameworks"].append("FastAPI")
                if "pygame" in content.lower():
                    self.knowledge_graph["frameworks"].append("Pygame")
                if "mediapipe" in content.lower():
                    self.knowledge_graph["frameworks"].append("MediaPipe")
                if "opencv" in content.lower():
                    self.knowledge_graph["frameworks"].append("OpenCV")

    def _scan_python_files(self):
        for root, dirs, files in os.walk(self.root_dir):
            dirs[:] = [d for d in dirs if d not in self.exclude_dirs]
            for file in files:
                filepath = Path(root) / file
                if self.is_python_file(filepath):
                    self._analyze_python_file(filepath)

    def _analyze_python_file(self, filepath: Path):
        try:
            with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
            tree = ast.parse(content)

            module_name = filepath.relative_to(self.root_dir).as_posix()
            imports = []
            functions = []
            classes = []

            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.append(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        imports.append(node.module)
                elif isinstance(node, ast.FunctionDef):
                    functions.append(node.name)
                elif isinstance(node, ast.ClassDef):
                    classes.append(node.name)

            self.knowledge_graph["modules"][module_name] = {
                "imports": list(set(imports)),
                "functions": functions,
                "classes": classes,
            }

            # Detect APIs or Endpoints
            if (
                "app.route" in content
                or "@app.get" in content
                or "@app.post" in content
                or "@app.websocket" in content
            ):
                self.knowledge_graph["apis"].append(module_name)
        except SyntaxError:
            pass
        except Exception as e:
            print(f"Error analyzing {filepath}: {e}")

    def _detect_environment(self):
        env_file = self.root_dir / ".env.example"
        if not env_file.exists():
            env_file = self.root_dir / ".env.bak"

        if env_file.exists():
            with open(env_file, "r", encoding="utf-8", errors="ignore") as f:
                lines = f.readlines()
                vars_list = [
                    line.split("=")[0].strip()
                    for line in lines
                    if "=" in line and not line.startswith("#")
                ]
                self.knowledge_graph["environment_variables"] = vars_list

    def _save_graph(self):
        output_file = self.root_dir / "knowledge_graph.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(self.knowledge_graph, f, indent=4)
        print(f"Knowledge graph saved to {output_file}")


if __name__ == "__main__":
    analyzer = RepoAnalyzer(".")
    analyzer.analyze()
