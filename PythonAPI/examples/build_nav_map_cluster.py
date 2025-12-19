#!/usr/bin/env python3
import argparse, json, math
from pathlib import Path

import carla

# 可选依赖：scikit-learn（更好的聚类）
try:
    from sklearn.cluster import DBSCAN
    HAVE_SK = True
except Exception:
    HAVE_SK = False


def d2(a, b):
    dx, dy = a[0] - b[0], a[1] - b[1]
    return (dx * dx + dy * dy) ** 0.5


def sample_waypoints(cmap, step):
    return list(cmap.generate_waypoints(step))


def junction_points(wps):
    pts = []
    for wp in wps:
        if wp.is_junction:
            loc = wp.transform.location
            pts.append((loc.x, loc.y, loc.z))
    return pts


def cluster_to_nodes(points, eps=20.0, min_samples=3, min_cluster_size=3):
    """
    把路口上的大量点聚成“每个路口一个质心节点”。
    - eps: 聚类半径（米）
    - min_samples: DBSCAN 参数
    - min_cluster_size: 小于这个点数的簇视为噪点丢弃
    返回 nodes: {id: {id,x,y,z}}, labels: len(points) 聚类标签（-1 噪点）
    """
    if not points:
        return {}, []

    centers = []
    labels = []

    if HAVE_SK:
        import numpy as np
        X = np.array([[p[0], p[1]] for p in points], dtype=float)
        db = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
        labels = db.labels_
        centers_raw = []
        for lbl in sorted(set(labels)):
            if lbl == -1:
                continue
            idxs = [i for i, v in enumerate(labels) if v == lbl]
            if len(idxs) < min_cluster_size:
                # 将这一簇标记为噪点
                for i in idxs:
                    labels[i] = -1
                continue
            xs = [points[i][0] for i in idxs]
            ys = [points[i][1] for i in idxs]
            zs = [points[i][2] for i in idxs]
            centers_raw.append((sum(xs) / len(xs), sum(ys) / len(ys), sum(zs) / len(zs)))
        centers = centers_raw
    else:
        # 简易贪心聚类（无 sklearn 也能跑）
        remaining = list(range(len(points)))
        labels = [-1] * len(points)
        cid = 0
        centers_raw = []
        while remaining:
            seed = remaining.pop()
            sx, sy = points[seed][0], points[seed][1]
            cluster_idx = [seed]
            keep = []
            for j in remaining:
                if d2((points[j][0], points[j][1]), (sx, sy)) <= eps:
                    cluster_idx.append(j)
                else:
                    keep.append(j)
            remaining = keep
            if len(cluster_idx) < min_cluster_size:
                # 小簇丢弃为噪点
                continue
            xs = [points[k][0] for k in cluster_idx]
            ys = [points[k][1] for k in cluster_idx]
            zs = [points[k][2] if len(points[k]) > 2 else 0.0 for k in cluster_idx]
            cx, cy = sum(xs) / len(xs), sum(ys) / len(ys)
            cz = sum(zs) / len(zs) if zs else 0.0
            centers_raw.append((cx, cy, cz))
            for k in cluster_idx:
                labels[k] = cid
            cid += 1
        centers = centers_raw

    nodes = {i: {"id": i, "x": c[0], "y": c[1], "z": c[2]} for i, c in enumerate(centers)}
    return nodes, labels


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
        # 近似到达结束：接近 end_wp
        if d2(
            (cur.transform.location.x, cur.transform.location.y),
            (end_wp.transform.location.x, end_wp.transform.location.y),
        ) <= step * 1.5:
            break
    coords = [(p.transform.location.x, p.transform.location.y, p.transform.location.z) for p in line]
    # 去重压缩（防止过密）
    out = []
    for c in coords:
        if not out or d2(out[-1][:2], c[:2]) >= step * 0.5:
            out.append(c)
    return out


def build_edges(cmap, nodes, step, snap_max=25.0):
    """
    用 map.get_topology() 得到“路段入口→出口”配对，映射到就近路口节点，
    合并双向与并行车道，最终每对路口节点只留一条线。
    - snap_max: 入口/出口吸附到路口节点的最大距离（米），超出则忽略该边，避免误连
    """
    topo = cmap.get_topology()
    merged = {}  # key=(min(u,v),max(u,v)) -> representative edge

    for (wp_a, wp_b) in topo:
        a = wp_a.transform.location
        b = wp_b.transform.location
        # 找最近路口节点并检查吸附距离
        ua = nearest_node(nodes, (a.x, a.y))
        ub = nearest_node(nodes, (b.x, b.y))
        da = d2((nodes[ua]["x"], nodes[ua]["y"]), (a.x, a.y))
        db = d2((nodes[ub]["x"], nodes[ub]["y"]), (b.x, b.y))
        if da > snap_max or db > snap_max:
            continue  # 太远说明这段并不直接连接两个路口，或路口未被正确聚类到

        u, v = ua, ub
        if u == v:
            continue

        coords = trace_lane_polyline(wp_a, wp_b, step)
        length = 0.0
        for i in range(1, len(coords)):
            length += d2(coords[i - 1], coords[i])

        key = (min(u, v), max(u, v))
        if key not in merged:
            merged[key] = {
                "u": key[0],
                "v": key[1],
                "length_m": length,
                "coords": coords,
            }
        else:
            # 已存在 u-v：保留更“代表性”的那条（更长/更完整）
            if length > merged[key]["length_m"]:
                merged[key]["length_m"] = length
                merged[key]["coords"] = coords

    return list(merged.values())


def write_geojson_nodes(nodes, path):
    feats = []
    for nid, a in nodes.items():
        feats.append(
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [a["x"], a["y"], a["z"]]},
                "properties": {"id": nid},
            }
        )
    fc = {"type": "FeatureCollection", "features": feats}
    Path(path).write_text(json.dumps(fc, ensure_ascii=False), encoding="utf-8")


def write_geojson_edges(edges, path):
    feats = []
    for e in edges:
        feats.append(
            {
                "type": "Feature",
                "geometry": {"type": "LineString", "coordinates": [[x, y, z] for (x, y, z) in e["coords"]]},
                "properties": {"u": e["u"], "v": e["v"], "length_m": round(e["length_m"], 3)},
            }
        )
    fc = {"type": "FeatureCollection", "features": feats}
    Path(path).write_text(json.dumps(fc, ensure_ascii=False), encoding="utf-8")


def write_compact_graph_json(nodes, edges, path):
    out = {
        "nodes": [{"id": int(nid), "x": a["x"], "y": a["y"], "z": a["z"]} for nid, a in nodes.items()],
        "edges": [{"u": e["u"], "v": e["v"], "length_m": e["length_m"], "polyline": e["coords"]} for e in edges],
    }
    Path(path).write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")


def maybe_plot(points, nodes, edges, out_prefix):
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    # 原始路口散点
    if points:
        plt.figure(figsize=(6, 6))
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        plt.scatter(xs, ys, s=6, alpha=0.5)
        plt.title("Raw junction waypoints")
        plt.axis("equal")
        plt.tight_layout()
        plt.savefig(str(out_prefix) + "_raw.png", dpi=180)
        plt.close()

    # 聚合后的路口 + 合并后的边
    plt.figure(figsize=(6, 6))
    if edges:
        for e in edges:
            xs = [c[0] for c in e["coords"]]
            ys = [c[1] for c in e["coords"]]
            plt.plot(xs, ys, lw=0.8, color="black", alpha=0.9)
    if nodes:
        xs = [a["x"] for a in nodes.values()]
        ys = [a["y"] for a in nodes.values()]
        plt.scatter(xs, ys, s=18, color="tab:blue", zorder=3)
    plt.title("Condensed junction graph")
    plt.axis("equal")
    plt.tight_layout()
    plt.savefig(str(out_prefix) + "_condensed.png", dpi=180)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=2000)
    ap.add_argument("--timeout", type=float, default=5.0)
    ap.add_argument("--step", type=float, default=2.0, help="车道中心采样间距（米）")
    ap.add_argument("--eps", type=float, default=10.0, help="路口聚类半径（米）")
    ap.add_argument("--min-samples", type=int, default=3, help="DBSCAN 最小样本数")
    ap.add_argument("--min-cluster-size", type=int, default=3, help="簇最小尺寸，小于此视为噪点")
    ap.add_argument("--snap", type=float, default=25.0, help="入口/出口吸附到路口节点的最大距离（米）")
    ap.add_argument("--out", default="nav", help="输出文件前缀")
    ap.add_argument("--plot", action="store_true", help="保存可视化 PNG")
    args = ap.parse_args()

    client = carla.Client(args.host, args.port)
    client.set_timeout(args.timeout)
    world = client.get_world()
    cmap = world.get_map()
    print(f"Map: {cmap.name}")

    wps = sample_waypoints(cmap, args.step)
    jpts = junction_points(wps)
    print(f"Raw junction waypoint count: {len(jpts)}")

    nodes, labels = cluster_to_nodes(
        jpts,
        eps=args.eps,
        min_samples=args.min_samples,
        min_cluster_size=args.min_cluster_size,
    )
    print(f"Junction nodes (clusters kept): {len(nodes)}")

    edges = build_edges(cmap, nodes, step=args.step, snap_max=args.snap)
    print(f"Undirected edges: {len(edges)}")

    prefix = Path(args.out)
    write_geojson_nodes(nodes, prefix.with_name(prefix.name + "_nodes.geojson"))
    write_geojson_edges(edges, prefix.with_name(prefix.name + "_edges.geojson"))
    write_compact_graph_json(nodes, edges, prefix.with_name(prefix.name + "_graph.json"))

    if args.plot:
        maybe_plot(jpts, nodes, edges, prefix)

    print(
        "Done:",
        prefix.with_name(prefix.name + "_nodes.geojson"),
        prefix.with_name(prefix.name + "_edges.geojson"),
        prefix.with_name(prefix.name + "_graph.json"),
        "(+ PNGs if --plot)",
    )


if __name__ == "__main__":
    main()
