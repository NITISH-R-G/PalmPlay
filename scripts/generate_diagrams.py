import json
import os
from pathlib import Path
import networkx as nx


def generate_interactive_html(graph: dict) -> str:
    """Generates an interactive D3.js visualization of the repository graph."""

    nodes = []
    links = []

    # Track node IDs to avoid duplicate errors in D3
    node_set = set()

    # Frameworks
    if "frameworks" in graph and graph["frameworks"]:
        for fw in set(graph["frameworks"]):
            node_id = f"fw_{fw}"
            if node_id not in node_set:
                nodes.append({"id": node_id, "group": 1, "label": fw})
                node_set.add(node_id)

    # Modules
    for module_name, details in graph.get("modules", {}).items():
        if module_name not in node_set:
            nodes.append({"id": module_name, "group": 2, "label": module_name})
            node_set.add(module_name)

        # Link internal imports
        for imp in details.get("imports", []):
            imp_node = imp.replace(".", "/") + ".py"

            # Simple matching: if import looks like it resolves to an internal module
            matched_module = None
            for mod in graph.get("modules", {}).keys():
                if imp in mod or mod.startswith(imp):
                    matched_module = mod
                    break

            if matched_module:
                links.append(
                    {"source": module_name, "target": matched_module, "value": 1}
                )

    # APIs
    if "apis" in graph:
        client_id = "Client"
        if client_id not in node_set:
            nodes.append({"id": client_id, "group": 3, "label": "API Client"})
            node_set.add(client_id)

        for api_mod in set(graph["apis"]):
            if api_mod in node_set:
                links.append({"source": client_id, "target": api_mod, "value": 2})

    graph_data = {"nodes": nodes, "links": links}
    graph_json = json.dumps(graph_data)

    html_template = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Interactive Architecture Diagram</title>
    <script src="https://d3js.org/d3.v7.min.js"></script>
    <style>
        body {{ margin: 0; padding: 0; overflow: hidden; background-color: #f4f4f9; font-family: sans-serif; }}
        .node circle {{ stroke: #fff; stroke-width: 1.5px; }}
        .node text {{ pointer-events: none; font-size: 12px; }}
        .link {{ stroke: #999; stroke-opacity: 0.6; }}
    </style>
</head>
<body>
    <div id="graph"></div>
    <script>
        const graphData = {graph_json};

        const width = window.innerWidth;
        const height = window.innerHeight;

        const color = d3.scaleOrdinal(d3.schemeCategory10);

        const svg = d3.select("#graph").append("svg")
            .attr("width", width)
            .attr("height", height)
            .call(d3.zoom().on("zoom", (event) => {{
               svg.attr("transform", event.transform);
            }}))
            .append("g");

        const simulation = d3.forceSimulation()
            .force("link", d3.forceLink().id(d => d.id).distance(100))
            .force("charge", d3.forceManyBody().strength(-300))
            .force("center", d3.forceCenter(width / 2, height / 2));

        const link = svg.append("g")
            .attr("class", "links")
            .selectAll("line")
            .data(graphData.links)
            .enter().append("line")
            .attr("class", "link")
            .attr("stroke-width", d => Math.sqrt(d.value));

        const node = svg.append("g")
            .attr("class", "nodes")
            .selectAll("g")
            .data(graphData.nodes)
            .enter().append("g")
            .call(d3.drag()
                .on("start", dragstarted)
                .on("drag", dragged)
                .on("end", dragended));

        node.append("circle")
            .attr("r", 10)
            .attr("fill", d => color(d.group));

        node.append("text")
            .attr("dx", 12)
            .attr("dy", ".35em")
            .text(d => d.label);

        // Click interaction for deep linking
        node.on("click", (event, d) => {{
            if(d.group === 2) {{
                // Try to open the file in the repo (assuming standard GitHub format if hosted)
                console.log("Clicked file:", d.id);
                // window.open("https://github.com/user/repo/blob/main/" + d.id, "_blank");
            }}
        }});

        simulation
            .nodes(graphData.nodes)
            .on("tick", ticked);

        simulation.force("link")
            .links(graphData.links);

        function ticked() {{
            link
                .attr("x1", d => d.source.x)
                .attr("y1", d => d.source.y)
                .attr("x2", d => d.target.x)
                .attr("y2", d => d.target.y);

            node
                .attr("transform", d => `translate(${{d.x}},${{d.y}})`);
        }}

        function dragstarted(event, d) {{
            if (!event.active) simulation.alphaTarget(0.3).restart();
            d.fx = d.x;
            d.fy = d.y;
        }}

        function dragged(event, d) {{
            d.fx = event.x;
            d.fy = event.y;
        }}

        function dragended(event, d) {{
            if (!event.active) simulation.alphaTarget(0);
            d.fx = null;
            d.fy = null;
        }}
    </script>
</body>
</html>
"""
    return html_template


def generate_mermaid_architecture(graph: dict) -> str:
    # Existing mermaid generator for backwards compatibility or dual-view
    mermaid = ["graph TD;"]

    if "frameworks" in graph and graph["frameworks"]:
        for fw in graph["frameworks"]:
            mermaid.append(f"    fw_{fw.replace(' ', '_')}:::framework;")
            mermaid.append(
                f"    classDef framework fill:#f9f,stroke:#333,stroke-width:2px;"
            )

    for module_name, details in graph.get("modules", {}).items():
        node_name = module_name.replace(".py", "").replace("/", "_")
        mermaid.append(f'    {node_name}["{module_name}"];')

        for imp in details.get("imports", []):
            imp_node = imp.replace(".", "_")
            if any(imp in mod for mod in graph.get("modules", {}).keys()):
                mermaid.append(f"    {node_name} --> {imp_node};")

    if "apis" in graph:
        for api_mod in graph["apis"]:
            api_node = api_mod.replace(".py", "").replace("/", "_")
            mermaid.append(f"    Client --> {api_node};")

    return "\n".join(mermaid)


def main():
    graph_path = Path("knowledge_graph.json")
    if not graph_path.exists():
        print("knowledge_graph.json not found. Run analyze_repo.py first.")
        return

    with open(graph_path, "r", encoding="utf-8") as f:
        graph = json.load(f)

    # Generate Mermaid
    architecture_diagram = generate_mermaid_architecture(graph)
    diagrams_dir = Path("diagrams")
    diagrams_dir.mkdir(exist_ok=True)
    with open(diagrams_dir / "architecture.mermaid", "w", encoding="utf-8") as f:
        f.write(architecture_diagram)

    # Generate Interactive HTML/SVG (D3.js)
    interactive_html = generate_interactive_html(graph)
    with open(
        diagrams_dir / "interactive_architecture.html", "w", encoding="utf-8"
    ) as f:
        f.write(interactive_html)

    print(
        f"Generated static architecture diagram at {diagrams_dir / 'architecture.mermaid'}"
    )
    print(
        f"Generated interactive SVG/HTML diagram at {diagrams_dir / 'interactive_architecture.html'}"
    )


if __name__ == "__main__":
    main()
