#!/usr/bin/env python3
import argparse, json, math
from pathlib import Path

import carla

# 可选依赖：networkx（若要图算法/GraphML），scikit-learn（更好的聚类）
try:
    from sklearn.cluster import DBSCAN
    HAVE_SK = True
except Exception:
    HAVE_SK = False


def d2(a, b):
    dx, dy = a[0]-b[0], a[1]-b[1]
    return (dx*dx + dy*dy) ** 0.5


def sample_waypoints(cmap, step):
    return list(cmap.generate_waypoints(step))


def junction_points(wps):
    pts = []
    for wp in wps:
        if wp.is_junction:
            loc = wp.transform.location
            pts.append((loc.x, loc.y, loc.z))
    return pts


def cluster_to_nodes(points, eps=10.0, min_samples=3):
    """把大量属于路口的点聚成每个路口一个质心节点。"""
    if not points:
        return {}
    if HAVE_SK:
        import numpy as np
        X = np.array([[p[0], p[1]] for p in points], dtype=float)
        db = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
        labels = db.labels_
        centers = []
        for lbl in sorted(set(labels)):
            if lbl == -1:
                continue
            idxs = [i for i, v in enumerate(labels) if v == lbl]
            xs = [points[i][0] for i in idxs]
            ys = [points[i][1] for i in idxs]
            zs = [points[i][2] for i in idxs]
            centers.append((sum(xs)/len(xs), sum(ys)/len(ys), sum(zs)/len(zs)))
    else:
        # 简易贪心聚类（没有 sklearn 也能跑）
        remaining = points[:]
        centers = []
        while remaining:
            x0, y0, z0 = remaining.pop()
            group = [(x0, y0, z0)]
            keep = []
            for (x, y, z) in remaining:
                if d2((x, y), (x0, y0)) <= eps:
                    group.append((x, y, z))
                else:
                    keep.append((x, y, z))
            remaining = keep
            cx = sum(p[0] for p in group)/len(group)
            cy = sum(p[1] for p in group)/len(group)
            cz = sum(p[2] for p in group)/len(group)
            centers.append((cx, cy, cz))
    # 编号
    return {i: {"id": i, "x": c[0], "y": c[1], "z": c[2]} for i, c in enumerate(centers)}


def nearest_node(nodes, xy):
    return min(nodes.keys(), key=lambda nid: d2((nodes[nid]["x"], nodes[nid]["y"]), xy))


def trace_lane_polyline(start_wp, end_wp, step):
    """沿 lane 从 start 走到 end，收集折线（尽量覆盖一整条路段）。"""
    line = [start_wp]
    cur = start_wp
    guard = 0
    while True:
        nxts = cur.next(step)
        if not nxts:
            break
        nxt = nxts[0]
        line.append(nxt)
        cur = nxt
        guard += 1
        if guard > 20000:
            break
        # 近似到达结束：接近 end_wp 同一位置/同一 road 段
        if d2(
            (cur.transform.location.x, cur.transform.location.y),
            (end_wp.transform.location.x, end_wp.transform.location.y)
        ) <= step * 1.5:
            break
    coords = [(p.transform.location.x, p.transform.location.y, p.transform.location.z) for p in line]
    # 去重压缩（防止过密）
    out = []
    for c in coords:
        if not out or d2(out[-1][:2], c[:2]) >= step*0.5:
            out.append(c)
    return out


def build_edges(cmap, nodes, step):
    """
    用 map.get_topology() 得到“路段入口→出口”配对，映射到就近路口节点，
    合并双向与并行车道，最终每对路口节点只留一条线。
    """
    topo = cmap.get_topology()
    merged = {}  # key=(min(u,v),max(u,v)) -> representative edge

    for (wp_a, wp_b) in topo:
        # 入口/出口 waypoint 可能离路口边界很近，直接投到最近的路口节点
        a = wp_a.transform.location
        b = wp_b.transform.location
        u = nearest_node(nodes, (a.x, a.y))
        v = nearest_node(nodes, (b.x, b.y))
        if u == v:
            continue

        # 从 a 走到 b 收集折线；另一方向也会在 topo 中出现，稍后用 key 合并
        coords = trace_lane_polyline(wp_a, wp_b, step)
        length = 0.0
        for i in range(1, len(coords)):
            length += d2(coords[i-1], coords[i])

        key = (min(u, v), max(u, v))
        if key not in merged:
            merged[key] = {
                "u": key[0], "v": key[1],
                "length_m": length,
                "coords": coords
            }
        else:
            # 已经存在一条 u-v：保留更“代表性”的那条（更长/更完整）
            if length > merged[key]["length_m"]:
                merged[key]["length_m"] = length
                merged[key]["coords"] = coords

    # 输出为列表
    edges = []
    for key, e in merged.items():
        edges.append(e)
    return edges


def write_geojson_nodes(nodes, path):
    feats = []
    for nid, a in nodes.items():
        feats.append({
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [a["x"], a["y"], a["z"]]},
            "properties": {"id": nid}
        })
    fc = {"type": "FeatureCollection", "features": feats}
    Path(path).write_text(json.dumps(fc, ensure_ascii=False), encoding="utf-8")


def write_geojson_edges(edges, nodes, path):
    feats = []
    for e in edges:
        feats.append({
            "type": "Feature",
            "geometry": {"type": "LineString", "coordinates": [[x, y, z] for (x, y, z) in e["coords"]]},
            "properties": {
                "u": e["u"], "v": e["v"],
                "length_m": round(e["length_m"], 3)
            }
        })
    fc = {"type": "FeatureCollection", "features": feats}
    Path(path).write_text(json.dumps(fc, ensure_ascii=False), encoding="utf-8")


def write_compact_graph_json(nodes, edges, path):
    out = {
        "nodes": [{"id": int(nid), "x": a["x"], "y": a["y"], "z": a["z"]} for nid, a in nodes.items()],
        "edges": [{"u": e["u"], "v": e["v"], "length_m": e["length_m"], "polyline": e["coords"]} for e in edges]
    }
    Path(path).write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=2000)
    ap.add_argument("--timeout", type=float, default=5.0)
    ap.add_argument("--step", type=float, default=2.0, help="车道中心采样间距（米）")
    ap.add_argument("--eps", type=float, default=10.0, help="路口聚类半径（米）")
    ap.add_argument("--min-samples", type=int, default=3, help="DBSCAN 最小样本数")
    ap.add_argument("--out", default="nav", help="输出文件前缀")
    args = ap.parse_args()

    client = carla.Client(args.host, args.port)
    client.set_timeout(args.timeout)
    world = client.get_world()
    cmap = world.get_map()
    print(f"Map: {cmap.name}")

    wps = sample_waypoints(cmap, args.step)
    jpts = junction_points(wps)
    nodes = cluster_to_nodes(jpts, eps=args.eps, min_samples=args.min_samples)
    print(f"Junction nodes: {len(nodes)}")

    edges = build_edges(cmap, nodes, step=args.step)
    print(f"Undirected edges: {len(edges)}")

    prefix = Path(args.out)
    write_geojson_nodes(nodes, prefix.with_name(prefix.name + "_nodes.geojson"))
    write_geojson_edges(edges, nodes, prefix.with_name(prefix.name + "_edges.geojson"))
    write_compact_graph_json(nodes, edges, prefix.with_name(prefix.name + "_graph.json"))
    print("Done:",
          prefix.with_name(prefix.name + "_nodes.geojson"),
          prefix.with_name(prefix.name + "_edges.geojson"),
          prefix.with_name(prefix.name + "_graph.json"))


if __name__ == "__main__":
    main()
