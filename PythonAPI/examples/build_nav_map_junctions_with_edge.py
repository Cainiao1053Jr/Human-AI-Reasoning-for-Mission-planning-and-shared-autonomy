import json
from pathlib import Path

in_path = Path("/path/to/your_points.geojson")
out_path = Path("/path/to/your_points_with_edges.geojson")

with open(in_path, "r", encoding="utf-8") as f:
    gj = json.load(f)

# 1) 建立 junction_id -> (x,y) 查找表
id2pt = {}
for ft in gj.get("features", []):
    geom = (ft.get("geometry") or {}).get("type")
    if geom != "Point":
        continue
    props = ft.get("properties") or {}
    jid = props.get("junction_id")
    coords = ft["geometry"]["coordinates"]
    if isinstance(jid, int) and len(coords) >= 2:
        id2pt[jid] = coords[:2]

# 2) 你想要连的“可用连接”列表（示例）
edges = [
    (101, 102),
    (102, 103),
    # ...继续填 (u, v)
]

# 3) 追加 LineString 边要素
for (u, v) in edges:
    if u not in id2pt or v not in id2pt:
        print(f"[WARN] skip ({u},{v})，找不到其中某个 junction_id")
        continue
    coords = [id2pt[u], id2pt[v]]
    gj["features"].append({
        "type": "Feature",
        "geometry": {"type":"LineString", "coordinates": coords},
        "properties": {"from": u, "to": v, "bidirectional": True}
    })

with open(out_path, "w", encoding="utf-8") as f:
    json.dump(gj, f, ensure_ascii=False, indent=2)

print(f"[OK] 写入：{out_path}")
