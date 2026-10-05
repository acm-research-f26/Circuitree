from collections import Counter
g = load_bench("c6288.bench")
print(Counter(d["gate"] for _, d in g.nodes(data=True) if d.get("gate") not in (None, "INPUT")))