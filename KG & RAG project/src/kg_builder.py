"""
Xây dựng Knowledge Graph cho 2 lĩnh vực: Tiểu đường và Tài chính.
Sử dụng NetworkX + Matplotlib để vẽ.
"""
import networkx as nx
import matplotlib.pyplot as plt
from config import KG_DIR


# =========================================================
# 1. KNOWLEDGE GRAPH CHO BỆNH TIỂU ĐƯỜNG
# =========================================================
def build_diabetes_kg() -> nx.DiGraph:
    """
    Đồ thị tri thức y tế về bệnh tiểu đường.
    Nodes: Bệnh nhân, Chỉ số, Nguy cơ, Biến chứng, Điều trị
    """
    G = nx.DiGraph()

    # --- Nhóm 1: Chỉ số (Indicators) ---
    indicators = {
        "Glucose": "mg/dL",
        "BMI": "kg/m2",
        "Age": "năm",
        "BloodPressure": "mmHg",
        "Insulin": "mu U/ml",
        "DiabetesPedigreeFunction": "score",
    }

    # --- Nhóm 2: Nguy cơ (Risks) ---
    risks = ["Béo phì", "Kháng insulin", "Tiền sử gia đình", "Tuổi cao"]

    # --- Nhóm 3: Bệnh (Diseases) ---
    diseases = ["Tiểu đường type 2", "Tiểu đường type 1"]

    # --- Nhóm 4: Biến chứng ---
    complications = ["Tim mạch", "Suy thận", "Mù lòa", "Loét chân"]

    # --- Nhóm 5: Điều trị ---
    treatments = ["Metformin", "Insulin", "Chế độ ăn", "Tập thể dục"]

    # Thêm nodes với thuộc tính nhóm để tô màu
    for name in indicators:
        G.add_node(name, group="indicator")
    for name in risks:
        G.add_node(name, group="risk")
    for name in diseases:
        G.add_node(name, group="disease")
    for name in complications:
        G.add_node(name, group="complication")
    for name in treatments:
        G.add_node(name, group="treatment")

    # --- Thêm edges (quan hệ) ---
    # Chỉ số → Nguy cơ
    G.add_edge("Glucose", "Kháng insulin", label="tăng")
    G.add_edge("BMI", "Béo phì", label=">= 30")
    G.add_edge("Age", "Tuổi cao", label=">= 45")
    G.add_edge("DiabetesPedigreeFunction", "Tiền sử gia đình", label="> 0.5")
    G.add_edge("BloodPressure", "Tim mạch", label="tăng")

    # Nguy cơ → Bệnh
    G.add_edge("Béo phì", "Tiểu đường type 2", label="dẫn đến")
    G.add_edge("Kháng insulin", "Tiểu đường type 2", label="dẫn đến")
    G.add_edge("Tiền sử gia đình", "Tiểu đường type 2", label="tăng nguy cơ")
    G.add_edge("Tuổi cao", "Tiểu đường type 2", label="tăng nguy cơ")
    G.add_edge("Kháng insulin", "Tiểu đường type 1", label="hiếm")

    # Bệnh → Biến chứng
    for comp in complications:
        G.add_edge("Tiểu đường type 2", comp, label="gây ra")

    # Điều trị → Bệnh
    for tx in treatments:
        G.add_edge(tx, "Tiểu đường type 2", label="điều trị")

    return G


# =========================================================
# 2. KNOWLEDGE GRAPH CHO TÀI CHÍNH
# =========================================================
def build_finance_kg() -> nx.DiGraph:
    """Đồ thị tri thức tài chính: Vàng, Chứng khoán, Vĩ mô."""
    G = nx.DiGraph()

    # --- Nodes ---
    assets = ["Vàng", "Chứng khoán", "USD", "Trái phiếu"]
    macro = ["Lạm phát", "Lãi suất Fed", "GDP", "Thất nghiệp", "Giá dầu"]
    strategies = ["Đầu tư dài hạn", "Lướt sóng", "Đa dạng hóa", "DCA"]

    for n in assets:  G.add_node(n, group="asset")
    for n in macro:   G.add_node(n, group="macro")
    for n in strategies: G.add_node(n, group="strategy")

    # --- Edges ---
    G.add_edge("Lạm phát", "Vàng", label="tăng → tăng")
    G.add_edge("Lãi suất Fed", "Vàng", label="tăng → giảm")
    G.add_edge("Lãi suất Fed", "USD", label="tăng → tăng")
    G.add_edge("USD", "Vàng", label="tương quan nghịch")
    G.add_edge("GDP", "Chứng khoán", label="tăng → tăng")
    G.add_edge("Thất nghiệp", "Chứng khoán", label="tăng → giảm")
    G.add_edge("Giá dầu", "Chứng khoán", label="tăng → giảm")
    G.add_edge("Lạm phát", "Trái phiếu", label="tăng → giảm")

    G.add_edge("Đa dạng hóa", "Vàng", label="phân bổ")
    G.add_edge("Đa dạng hóa", "Chứng khoán", label="phân bổ")
    G.add_edge("Đa dạng hóa", "Trái phiếu", label="phân bổ")
    G.add_edge("DCA", "Chứng khoán", label="chiến lược")
    G.add_edge("Đầu tư dài hạn", "Chứng khoán", label="chiến lược")

    return G


# =========================================================
# 3. VẼ ĐỒ THỊ
# =========================================================
COLOR_MAP = {
    "indicator": "#4C9BE8", "risk": "#E86A4C", "disease": "#C0392B",
    "complication": "#8E44AD", "treatment": "#27AE60",
    "asset": "#F1C40F", "macro": "#3498DB", "strategy": "#2ECC71",
}


def draw_kg(G: nx.DiGraph, title: str, filename: str, layout: str = "spring"):
    """Vẽ và lưu đồ thị tri thức."""
    plt.figure(figsize=(16, 10))

    if layout == "spring":
        pos = nx.spring_layout(G, k=1.5, seed=42, iterations=80)
    elif layout == "kamada":
        pos = nx.kamada_kawai_layout(G)
    else:
        pos = nx.shell_layout(G)

    node_colors = [COLOR_MAP.get(G.nodes[n].get("group", ""), "#95A5A6") for n in G.nodes]

    # Vẽ nodes
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=2200,
                           edgecolors="black", linewidths=1.2, alpha=0.9)
    # Vẽ labels
    nx.draw_networkx_labels(G, pos, font_size=10, font_weight="bold", font_color="white")
    # Vẽ edges
    nx.draw_networkx_edges(G, pos, edge_color="#7F8C8D", arrows=True,
                           arrowsize=15, width=1.5, connectionstyle="arc3,rad=0.1")
    # Vẽ edge labels
    edge_labels = nx.get_edge_attributes(G, "label")
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8, font_color="#E74C3C")

    plt.title(title, fontsize=16, fontweight="bold")
    plt.axis("off")
    plt.tight_layout()
    out = KG_DIR / filename
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[KG] Đã lưu hình: {out}")


# =========================================================
# 4. TRUY VẤN ĐỒ THỊ
# =========================================================
def query_neighbors(G: nx.DiGraph, node: str, depth: int = 1) -> dict:
    """Truy vấn các node lân cận trong bán kính `depth`."""
    if node not in G:
        return {"error": f"Node '{node}' không tồn tại trong đồ thị."}

    result = {"center": node, "neighbors": []}
    for n in nx.descendants_at_distance(G, node, depth):
        path = nx.shortest_path(G, node, n)
        result["neighbors"].append({"node": n, "path": path})
    return result


def graph_to_text(G: nx.DiGraph) -> str:
    """Chuyển đồ thị thành văn bản để đưa vào RAG context."""
    lines = []
    for u, v, data in G.edges(data=True):
        label = data.get("label", "liên quan đến")
        lines.append(f"- {u} {label} {v}")
    return "\n".join(lines)


# =========================================================
# 5. MAIN
# =========================================================
if __name__ == "__main__":
    print("=" * 60)
    print("XÂY DỰNG KNOWLEDGE GRAPH")
    print("=" * 60)

    # Diabetes KG
    G_diab = build_diabetes_kg()
    print(f"\n[Diabetes KG] Nodes: {G_diab.number_of_nodes()}, Edges: {G_diab.number_of_edges()}")
    draw_kg(G_diab, "Knowledge Graph: Bệnh Tiểu Đường",
            "kg_diabetes.png", layout="spring")

    # Finance KG
    G_fin = build_finance_kg()
    print(f"\n[Finance KG] Nodes: {G_fin.number_of_nodes()}, Edges: {G_fin.number_of_edges()}")
    draw_kg(G_fin, "Knowledge Graph: Tài Chính (Vàng & Chứng khoán)",
            "kg_finance.png", layout="spring")

    # Demo truy vấn
    print("\n[Demo truy vấn Diabetes KG]")
    print(query_neighbors(G_diab, "Glucose", depth=1))

    print("\n[Demo truy vấn Finance KG]")
    print(query_neighbors(G_fin, "Lạm phát", depth=1))