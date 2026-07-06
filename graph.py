# save_graph.py
from backend.rag_graph import build_graph

graph = build_graph()

# Save as PNG
png_bytes = graph.get_graph().draw_mermaid_png()
with open("assets/graph.png", "wb") as f:
    f.write(png_bytes)

print("Saved to assets/graph.png")