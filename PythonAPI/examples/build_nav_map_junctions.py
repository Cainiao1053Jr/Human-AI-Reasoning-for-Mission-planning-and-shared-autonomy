#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CARLA：仅路口点（每个路口一个代表点）
- 代表点 = 该路口内所有 waypoint(x,y,z) 的均值；
- 若某路口采不到点，则退回到其 bounding_box.center；
- 可选择画到 CARLA 世界；可导出为仅包含 Point 的 GeoJSON。
"""

import argparse
import json
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

import carla

# --------------------- 工具函数 ---------------------

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

# --------------------- 路口采样与代表点 ---------------------

def collect_junction_points(world: carla.World, sample_step: float = 2.0) -> Dict[int, List[Tuple[float, float, float]]]:
    """
    扫全图 waypoint，筛 is_junction==True，按 junction.id 分组收集 (x,y,z)。
    """
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

def build_junction_representatives(
    world: carla.World,
    sample_step: float = 2.0,
    use_bb_center_if_empty: bool = True
) -> Dict[int, Tuple[float, float, float]]:
    """
    返回每个 junction.id 的代表点坐标：
      - 优先用该路口内 waypoint 的均值；
      - 若某路口没有采到点，退回到其 bounding_box.center（可选）。
    """
    mp = world.get_map()
    grouped = collect_junction_points(world, sample_step=sample_step)

    # 预取所有路口的 bbox center 以兜底
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

# --------------------- 仅绘制“路口点” ---------------------

def draw_junction_points(world: carla.World,
                         junction_reps: Dict[int, Tuple[float, float, float]],
                         life_time: float = 20.0):
    dbg = world.debug
    for jid, (x, y, z) in junction_reps.items():
        loc = carla.Location(x=x, y=y, z=z + 0.5)
        dbg.draw_point(loc, size=0.14, life_time=life_time, persistent_lines=False)
        dbg.draw_string(loc + carla.Location(z=0.5), f"J{jid}", False, life_time=life_time)

# --------------------- 仅导出“路口点”为 GeoJSON ---------------------

def export_junction_points_geojson(path: str,
                                   junction_reps: Dict[int, Tuple[float, float, float]]):
    """
    导出为 GeoJSON（2D）：仅包含路口代表点（Point），不含任何线（LineString）。
    """
    features = []
    for jid, (x, y, z) in junction_reps.items():
        features.append({
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [x, y]},
            "properties": {"junction_id": jid, "z": z}
        })
    gj = {"type": "FeatureCollection", "features": features}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(gj, f, ensure_ascii=False, indent=2)
    print(f"[OK] GeoJSON (junction points only) written to {path}")

# --------------------- 主流程 ---------------------

def main():
    parser = argparse.ArgumentParser(description="CARLA junction representatives only (points).")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--timeout", type=float, default=5.0)

    parser.add_argument("--wp_step_node", type=float, default=2.0,
                        help="Waypoint step for junction node sampling.")
    parser.add_argument("--use_bb_center", action="store_true",
                        help="If set, use junction bbox center when no waypoint points were sampled.")
    parser.add_argument("--draw", action="store_true", help="Draw points into CARLA world.")
    parser.add_argument("--life", type=float, default=20.0, help="Debug draw life time.")
    parser.add_argument("--geojson", type=str, default="map2_junction_points.geojson",
                        help="Export GeoJSON path (points only).")
    args = parser.parse_args()

    client = carla.Client(args.host, args.port)
    client.set_timeout(args.timeout)
    world = client.get_world()

    print("[*] Building junction representatives (mean-by-id, points only)...")
    junction_reps = build_junction_representatives(
        world,
        sample_step=args.wp_step_node,
        use_bb_center_if_empty=args.use_bb_center
    )
    print(f"    -> {len(junction_reps)} junction points")

    if args.draw:
        print("[*] Drawing junction points in world...")
        draw_junction_points(world, junction_reps, life_time=args.life)

    if args.geojson:
        print("[*] Exporting GeoJSON (points only)...")
        export_junction_points_geojson(args.geojson, junction_reps)

    print("[Done]")

if __name__ == "__main__":
    main()
