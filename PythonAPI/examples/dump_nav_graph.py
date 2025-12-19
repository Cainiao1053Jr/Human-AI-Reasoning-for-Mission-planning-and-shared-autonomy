#!/usr/bin/env python3
import argparse, json, csv, math
from pathlib import Path
from collections import defaultdict

import carla

try:
    import networkx as nx
    HAVE_NX = True
except Exception:
    HAVE_NX = False

try:
    from sklearn.cluster import DBSCAN
    HAVE_SK = True
except Exception:
    HAVE_SK = False


def dist2d(a, b):
    dx, dy = a[0]-b[0], a[1]-b[1]
    return (dx*dx + dy*dy) ** 0.5


def sample_all_waypoints(carla_map, step):
    return list(carla_map.generate_waypoints(step))


def get_junction_points_from_waypoints(wps):
    pts = []
    for wp in wps:
        if wp.is_junction:
            loc = wp.transform.location
            pts.append((loc.x, loc.y, loc.z))
    return pts


def cluster_points(points, eps=10.0, min_samples=3):
    if not points:
        return [], {}
    if HAVE_SK:
        import numpy as np
        X = np.array([[p[0], p[1]] for p in points], dtype=float)
        db = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
        labels = db.labels_
        centers, assign = [], {}
        for lbl in sorted(set(labels)):
            if lbl == -1:
                continue
            idxs = [i for i, v in enumerate(labels) if v == lbl]
            xs = [points[i][0] for i in idxs]
            ys = [points[i][1] for i in idxs]
            zs = [points[i][2] for i in idxs]
            cx, cy, cz = sum(xs)/len(xs), sum(ys)/len(ys), sum(zs)/len(zs)
            centers.append((cx, cy, cz))
            for i in idxs:
                assign[i] = len(centers)-1
        return centers, assign
    # 无 sklearn：简易半径聚合
    remaining = points[:]
    centers, assign = [], {}
    used = [False]*len(points)
    for i,(x0,y0,z0) in enumerate(points):
        if used[i]:
            continue
        cluster = [(x0,y0,z0)]
        used[i] = True
        for j,(x,y,z) in enumerate(points):
            if used[j]:
                continue
            if dist2d((x,y),(x0,y0)) <= eps:
                cluster.append((x,y,z))
                used[j] = True
        cx = sum(p[0] for p in cluster)/len(cluster)
        cy = sum(p[1] for p in cluster)/len(cluster)
        cz = sum(p[2] for p in cluster)/len(cluster)
        centers.append((cx,cy,cz))
    # 赋值表粗略映射
    for i,p in enumerate(points):
        # 找最近中心
        k = min(range(len(centers)), key=lambda t: dist2d((p[0],p[1]), (centers[t][0],centers[t][1])))
        assign[i] = k
    return centers, assign


def build_node_index_from_junction_clusters(wps, eps=10.0, min_samples=3):
    jpts = get_junction_points_from_waypoints(wps)
    centers, _ = cluster_points(jpts, eps=eps, min_samples=min_samples)
    nodes = {i: {"id": i, "x": c[0], "y": c[1], "z": c[2]} for i, c in enumerate(centers)}
    return nodes


def nearest_node(nodes, loc_xy):
    return min(nodes.keys(), key=lambda nid: dist2d((nodes[nid]["x"], nodes[nid]["y"]), loc_xy))


def walk_to_next_junction(start_wp, step, is_junction):
    """从非路口 waypoint 沿车道走，直到到达路口或车道结束；返回折线点列与终点 wp。"""
    line = [start_wp]
    cur = start_wp
    guard = 0
    while True:
        nxts = cur.next(step)
        if not nxts:
            break
        cur = nxts[0]
        line.append(cur)
        if is_junction(cur):
            break
        guard += 1
        if guard > 5000:
            break
    return line, cur


def condense_to_junction_graph(carla_map, wps, nodes, step):
    """把路网压缩为：路口节点 + 路口间边（折线），并统计长度、限速均值、方向等。"""
    # 标记函数
    def is_junction_wp(wp): return bool(wp.is_junction)

    # 非路口车道入口（从“非路口”进入“路口”的方向，也可反过来）
    # 我们从所有非路口点里，挑在“接近路口邻域”且其 next 方向将进入路口的起点
    # 简化：直接从所有非路口点出发沿 lane 走到路口，记录起点最近路口与终点路口
    edges = []
    visited_hash = set()

    for wp in wps:
        if is_junction_wp(wp):
            continue
        # 以 (road, section, lane, s_idx) 粗略去重
        key = (wp.road_id, wp.section_id, wp.lane_id, int(wp.s if hasattr(wp, "s") else wp.transform.location.x*10))
        if key in visited_hash:
            continue
        poly, end_wp = walk_to_next_junction(wp, step, is_junction_wp)
        if len(poly) < 2 or not is_junction_wp(end_wp):
            continue

        # 连接的两个路口节点：起点所在最近路口（用“回看 step”找入端），终点路口
        start_loc = poly[0].transform.location
        end_loc = end_wp.transform.location
        # 若起点离某路口很近则投到该路口节点，否则暂不连起点路口（会导致边从“路段内点”起跑）
        # 简易：都投最近路口节点（若图稀疏，起点可能不在路口附近，问题不大）
        u = nearest_node(nodes, (start_loc.x, start_loc.y))
        v = nearest_node(nodes, (end_loc.x, end_loc.y))
        if u == v:
            continue

        # 构造折线坐标
        coords = [(p.transform.location.x, p.transform.location.y, p.transform.location.z) for p in poly]
        # 长度与限速统计
        length = 0.0
        speeds = []
        for i in range(1, len(coords)):
            length += dist2d(coords[i-1], coords[i])
        for p in poly:
            try:
                speeds.append(float(p.lane_width) * 0 + float(p.speed_limit))  # speed_limit 有些地图为 0
            except Exception:
                pass
        speed_mean = sum(speeds)/len(speeds) if speeds else 0.0

        edges.append({
            "u": u, "v": v,
            "length_m": length,
            "speed_limit_mean": speed_mean,
            "coords": coords
        })

    # 去重与合并（同一 u-v 可能多条并行车道）
    merged = {}
    for e in edges:
        key = (min(e["u"], e["v"]), max(e["u"], e["v"]))
        if key not in merged:
            merged[key] = {
                "u": key[0], "v": key[1],
                "length_m": e["length_m"],
                "count": 1,
                "speed_limit_mean": e["speed_limit_mean"],
                "coords_list": [e["coords"]]
            }
        else:
            m = merged[key]
            m["length_m"] = min(m["length_m"], e["length_m"])  # 取最短作为代表
            m["speed_limit_mean"] = (m["speed_limit_mean"]*m["count"] + e["speed_limit_mean"]) / (m["count"]+1)
            m["count"] += 1
            if len(e["coords"]) > len(m["coords_list"][0]):
                m["coords_list"][0] = e["coords"]  # 任选“看起来更完整”的折线做展示

    return list(merged.values())


def write_nodes_geojson(nodes, path):
    feats = []
    for nid, attr in nodes.items():
        feats.append({
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [attr["x"], attr["y"], attr["z"]]},
            "properties": {"id": nid}
        })
    fc = {"type": "FeatureCollection", "features": feats}
    Path(path).write_text(json.dumps(fc, ensure_ascii=False), encoding="utf-8")
    print(f"Nodes GeoJSON: {path} (N={len(nodes)})")


def write_edges_geojson(edges, path):
    feats = []
    for e in edges:
        feats.append({
            "type": "Feature",
            "geometry": {"type": "LineString",
                         "coordinates": [[x,y,z] for (x,y,z) in e["coords_list"][0]]},
            "properties": {
                "u": e["u"], "v": e["v"],
                "length_m": round(e["length_m"], 3),
                "lanes_parallel": e["count"],
                "speed_mean": round(e["speed_limit_mean"], 1)
            }
        })
    fc = {"type": "FeatureCollection", "features": feats}
    Path(path).write_text(json.dumps(fc, ensure_ascii=False), encoding="utf-8")
    print(f"Edges GeoJSON: {path} (E={len(edges)})")


def write_graphml(nodes, edges, path):
    if not HAVE_NX:
        print("networkx 未安装，跳过 GraphML（pip install networkx 可启用）")
        return
    G = nx.Graph()
    for nid, a in nodes.items():
        G.add_node(nid, x=a["x"], y=a["y"], z=a["z"])
    for e in edges:
        G.add_edge(e["u"], e["v"],
                   length_m=float(e["length_m"]),
                   lanes_parallel=int(e["count"]),
                   speed_mean=float(e["speed_limit_mean"]))
    nx.write_graphml(G, path)
    print(f"GraphML: {path} (|V|={len(nodes)}, |E|={len(edges)})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=2000)
    ap.add_argument("--timeout", type=float, default=5.0)
    ap.add_argument("--step", type=float, default=2.0)
    ap.add_argument("--eps", type=float, default=10.0)
    ap.add_argument("--min-samples", type=int, default=3)
    ap.add_argument("--out", default="nav_graph")
    args = ap.parse_args()

    client = carla.Client(args.host, args.port)
    client.set_timeout(args.timeout)
    world = client.get_world()
    carla_map = world.get_map()
    print(f"Map: {carla_map.name}")

    wps = sample_all_waypoints(carla_map, args.step)
    nodes = build_node_index_from_junction_clusters(wps, eps=args.eps, min_samples=args.min_samples)
    edges = condense_to_junction_graph(carla_map, wps, nodes, step=args.step)

    prefix = Path(args.out)
    write_nodes_geojson(nodes, prefix.with_name(prefix.name + "_nodes.geojson"))
    write_edges_geojson(edges, prefix.with_name(prefix.name + "_edges.geojson"))
    write_graphml(nodes, edges, prefix.with_suffix(".graphml"))


if __name__ == "__main__":
    main()
