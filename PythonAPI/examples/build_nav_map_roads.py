#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CARLA：路口(每个路口一个点=均值) + 路段(双向合并；折线端点贴在路口代表点上)

改动要点：
  - 采样边时从拓扑段中点(用 wp_a)向两侧各走：backward 到源路口、forward 到目的路口
  - 折线拼接后，将首尾“对齐”到两端路口代表点：直接把代表点插入到 poly 首尾
  - 做去重/去抖，避免重复点或极短段
"""

import argparse
import json
from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Set, FrozenSet

import carla
import math

# --------------------- 工具函数 ---------------------

def dist2(a: Tuple[float,float,float], b: Tuple[float,float,float]) -> float:
    dx, dy, dz = a[0]-b[0], a[1]-b[1], a[2]-b[2]
    return dx*dx + dy*dy + dz*dz

def loc_tuple(loc: carla.Location) -> Tuple[float, float, float]:
    return (float(loc.x), float(loc.y), float(loc.z))

def wp_xyz(wp: carla.Waypoint) -> Tuple[float, float, float]:
    return loc_tuple(wp.transform.location)

def average_xyz(points: List[Tuple[float, float, float]]) -> Tuple[float, float, float]:
    n = max(1, len(points))
    sx = sum(p[0] for p in points)
    sy = sum(p[1] for p in points)
    sz = sum(p[2] for p in points)
    return (sx / n, sy / n, sz / n)

def bb_center(bb: carla.BoundingBox) -> Tuple[float, float, float]:
    return loc_tuple(bb.location)

def dedupe_close(points: List[Tuple[float,float,float]], eps: float=0.05) -> List[Tuple[float,float,float]]:
    """移除连续的、彼此非常接近的点（阈值 eps 米）。"""
    if not points: return points
    out = [points[0]]
    eps2 = eps*eps
    for p in points[1:]:
        if dist2(p, out[-1]) > eps2:
            out.append(p)
    return out

# --------------------- 路口采样 ---------------------

def collect_junction_points(world: carla.World, sample_step: float = 2.0) -> Dict[int, List[Tuple[float, float, float]]]:
    """扫全图 waypoint，筛 is_junction==True，按 junction.id 分组收集 (x,y,z)"""
    mp = world.get_map()
    wps = mp.generate_waypoints(sample_step)
    junc_points: Dict[int, List[Tuple[float, float, float]]] = defaultdict(list)
    for wp in wps:
        if wp.is_junction:
            j = wp.get_junction()
            if j is None:
                continue
            junc_points[j.id].append(wp_xyz(wp))
    return junc_points

def build_junction_representatives(world: carla.World,
                                   sample_step: float = 2.0,
                                   use_bb_center_if_empty: bool = True) -> Dict[int, Tuple[float, float, float]]:
    """
    返回每个 junction.id 的代表点坐标（对该路口内所有 waypoint 平均）。
    若某路口采不到点，可选用其 bounding_box.center 兜底。
    """
    mp = world.get_map()
    grouped = collect_junction_points(world, sample_step=sample_step)

    bbox_centers: Dict[int, Tuple[float, float, float]] = {}
    for wp in mp.generate_waypoints(8.0):
        if wp.is_junction:
            j = wp.get_junction()
            if j and j.id not in bbox_centers:
                bbox_centers[j.id] = bb_center(j.bounding_box)

    reps: Dict[int, Tuple[float, float, float]] = {}
    for jid, pts in grouped.items():
        reps[jid] = average_xyz(pts)

    if use_bb_center_if_empty:
        for jid, center in bbox_centers.items():
            if jid not in reps:
                reps[jid] = center
    return reps

# --------------------- 端点路口识别/采样 ---------------------

def walk_until_junction_id(start_wp: carla.Waypoint,
                           step: float,
                           forward: bool,
                           max_iter: int = 1200) -> Optional[int]:
    """沿车道前/后走，直到进入路口，返回 junction.id；否则 None。"""
    wp = start_wp
    for _ in range(max_iter):
        if wp.is_junction:
            j = wp.get_junction()
            return None if j is None else j.id
        nxt = wp.next(step) if forward else wp.previous(step)
        if not nxt:
            break
        wp = nxt[0]
    return None

def sample_to_junction(start_wp: carla.Waypoint,
                       step: float,
                       forward: bool,
                       max_iter: int = 1200) -> List[Tuple[float,float,float]]:
    """
    从 start_wp 出发，沿 forward/ backward 方向采样 waypoint 坐标，
    直到进入路口或无路为止。包含起点。
    """
    pts: List[Tuple[float,float,float]] = [wp_xyz(start_wp)]
    wp = start_wp
    for _ in range(max_iter):
        if wp.is_junction:
            break
        nxt = wp.next(step) if forward else wp.previous(step)
        if not nxt:
            break
        wp = nxt[0]
        pts.append(wp_xyz(wp))
        if wp.is_junction:
            break
    return pts

# --------------------- 无向道路构建（端点对齐代表点） ---------------------

def build_undirected_edges_snap_to_reps(world: carla.World,
                                        junction_reps: Dict[int, Tuple[float, float, float]],
                                        travel_step: float = 2.0,
                                        sample_step: float = 2.0,
                                        max_walk_iter: int = 1200,
                                        snap_eps: float = 0.25
                                        ) -> Dict[FrozenSet[int], List[Tuple[float, float, float]]]:
    """
    基于 map.get_topology()：
      - 找到每段的两端路口 id (j_src, j_dst)
      - 从段内参考点(用 wp_a)向后采样到源路口、向前采样到目的路口，拼接为完整折线
      - 折线首尾强制对齐到 junction_reps[j_src]/junction_reps[j_dst]
      - 用 frozenset({j1,j2}) 合并双向

    返回：{frozenset({j1,j2}): [(x,y,z), ...]}
    """
    mp = world.get_map()
    topology = mp.get_topology()
    edges: Dict[FrozenSet[int], List[Tuple[float, float, float]]] = {}

    for (wp_a, wp_b) in topology:
        # 识别两端路口 id
        j_src = walk_until_junction_id(wp_a, step=travel_step, forward=False, max_iter=max_walk_iter)
        j_dst = walk_until_junction_id(wp_a, step=travel_step, forward=True,  max_iter=max_walk_iter)

        if j_src is None or j_dst is None or j_src == j_dst:
            continue
        if j_src not in junction_reps or j_dst not in junction_reps:
            # 没有代表点就跳过
            continue

        key = frozenset({j_src, j_dst})
        if key in edges:
            continue  # 另一方向已写

        # 采样两侧，拼接成从源->目的的折线
        back_part = sample_to_junction(wp_a, step=sample_step, forward=False, max_iter=max_walk_iter)
        fwd_part  = sample_to_junction(wp_a, step=sample_step, forward=True,  max_iter=max_walk_iter)

        # back_part 方向是从中点 -> 源路口，需要反转；并且避免重复中点
        if back_part:
            back_part = back_part[::-1]
        if back_part and fwd_part and dist2(back_part[-1], fwd_part[0]) < 1e-6:
            poly = back_part + fwd_part[1:]  # 去掉重复的跨点
        else:
            poly = (back_part or []) + (fwd_part or [])

        # 去除过密点
        poly = dedupe_close(poly, eps=0.05)
        if len(poly) < 2:
            continue

        # --- 关键：把首尾对齐到路口代表点 ---
        src_rep = junction_reps[j_src]
        dst_rep = junction_reps[j_dst]

        # 若首点离 src_rep 很近就替换，否则直接插入 src_rep
        if dist2(poly[0], src_rep) > (snap_eps * snap_eps):
            poly = [src_rep] + poly
        else:
            poly[0] = src_rep

        # 末点贴到 dst_rep
        if dist2(poly[-1], dst_rep) > (snap_eps * snap_eps):
            poly = poly + [dst_rep]
        else:
            poly[-1] = dst_rep

        # 再做一次去抖，防止首尾插入后连着两个重复点
        poly = dedupe_close(poly, eps=0.02)
        if len(poly) < 2:
            continue

        edges[key] = poly

    return edges

# --------------------- 可视化与导出 ---------------------

def draw_in_world(world: carla.World,
                  junction_reps: Dict[int, Tuple[float, float, float]],
                  edges: Dict[FrozenSet[int], List[Tuple[float, float, float]]],
                  life_time: float = 20.0):
    dbg = world.debug

    # 画路口点 + id
    for jid, (x, y, z) in junction_reps.items():
        loc = carla.Location(x=x, y=y, z=z + 0.5)
        dbg.draw_point(loc, size=0.14, life_time=life_time, persistent_lines=False)
        dbg.draw_string(loc + carla.Location(z=0.5), f"J{jid}", False, life_time=life_time)

    # 画折线（确保首尾就是路口代表点）
    for key, poly in edges.items():
        for i in range(len(poly) - 1):
            x1, y1, z1 = poly[i]
            x2, y2, z2 = poly[i + 1]
            dbg.draw_line(carla.Location(x1, y1, z1 + 0.1),
                          carla.Location(x2, y2, z2 + 0.1),
                          thickness=0.08,
                          life_time=life_time)

def export_geojson(path: str,
                   junction_reps: Dict[int, Tuple[float, float, float]],
                   edges: Dict[FrozenSet[int], List[Tuple[float, float, float]]]):
    """
    导出为 GeoJSON（2D），点是路口代表点，线是无向折线（首尾为路口点）。
    """
    features = []

    for jid, (x, y, z) in junction_reps.items():
        features.append({
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [x, y]},
            "properties": {"junction_id": jid}
        })

    for key, poly in edges.items():
        j1, j2 = list(key)
        coords2d = [[p[0], p[1]] for p in poly]
        features.append({
            "type": "Feature",
            "geometry": {"type": "LineString", "coordinates": coords2d},
            "properties": {"j1": j1, "j2": j2}
        })

    gj = {"type": "FeatureCollection", "features": features}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(gj, f, ensure_ascii=False, indent=2)
    print(f"[OK] GeoJSON written to {path}")

# --------------------- 主流程 ---------------------

def main():
    parser = argparse.ArgumentParser(description="CARLA junction nodes (averaged) + undirected road lines snapped to junction reps.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--timeout", type=float, default=5.0)
    parser.add_argument("--wp_step_node", type=float, default=2.0, help="Waypoint step for junction node sampling.")
    parser.add_argument("--travel_step", type=float, default=2.0, help="Step to walk to detect endpoint junctions.")
    parser.add_argument("--sample_step_edge", type=float, default=2.0, help="Polyline sampling step for edges.")
    parser.add_argument("--snap_eps", type=float, default=0.25, help="Snap tolerance (m) to decide whether to insert reps.")
    parser.add_argument("--draw", action="store_true", help="Draw into CARLA world.")
    parser.add_argument("--life", type=float, default=20.0, help="Debug draw life time.")
    parser.add_argument("--geojson", type=str, default="map2_points.geojson", help="Export GeoJSON path.")
    args = parser.parse_args()

    client = carla.Client(args.host, args.port)
    client.set_timeout(args.timeout)
    world = client.get_world()

    print("[*] Building junction representatives (mean-by-id)...")
    junction_reps = build_junction_representatives(
        world,
        sample_step=args.wp_step_node,
        use_bb_center_if_empty=True
    )
    print(f"    -> {len(junction_reps)} junction nodes")

    print("[*] Building undirected edges snapped to reps (merge both directions)...")
    edges = build_undirected_edges_snap_to_reps(
        world,
        junction_reps=junction_reps,
        travel_step=args.travel_step,
        sample_step=args.sample_step_edge,
        max_walk_iter=1500,
        snap_eps=args.snap_eps
    )
    print(f"    -> {len(edges)} undirected edges")

    if args.draw:
        print("[*] Drawing in world...")
        draw_in_world(world, junction_reps, edges, life_time=args.life)

    if args.geojson:
        print("[*] Exporting GeoJSON...")
        export_geojson(args.geojson, junction_reps, edges)

    print("[Done]")

if __name__ == "__main__":
    main()
