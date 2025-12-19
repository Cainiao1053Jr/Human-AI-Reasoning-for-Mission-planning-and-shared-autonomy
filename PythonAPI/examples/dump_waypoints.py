#!/usr/bin/env python3
# dump_waypoints.py

import argparse
import csv
import json
from pathlib import Path

import carla


def waypoint_rows_from_map(carla_map: carla.Map, step: float):
    """
    使用 map.generate_waypoints(step) 采样全图所有可行驶车道中心线。
    返回生成器，每行是一个 dict，包含常用字段。
    """
    wps = carla_map.generate_waypoints(step)  # 每隔 step 米一个点（所有车道）
    for wp in wps:
        tr = wp.transform
        loc = tr.location
        rot = tr.rotation
        yield {
            "road_id": wp.road_id,
            "section_id": wp.section_id,
            "lane_id": wp.lane_id,
            "lane_type": str(wp.lane_type),
            "is_junction": int(wp.is_junction),
            "x": loc.x,
            "y": loc.y,
            "z": loc.z,
            "yaw_deg": rot.yaw,
            "pitch_deg": rot.pitch,
            "roll_deg": rot.roll,
            "lane_width": wp.lane_width,
            # 下面这些有些地图可能无效或恒零，保留占位
            "lane_change": str(wp.lane_change),
        }


def write_csv(rows, csv_path: Path):
    rows = list(rows)
    if not rows:
        print("No waypoints generated.")
        return
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(rows[0].keys())
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"CSV written: {csv_path}  (rows={len(rows)})")


def write_geojson(rows, geojson_path: Path):
    """
    简单把每个 waypoint 导成一个 Point（非线），
    方便快速可视化。若想导出“折线”，需要先按
    road_id/section_id/lane_id 分组，按空间顺序连线。
    """
    features = []
    for r in rows:
        features.append({
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [r["x"], r["y"], r["z"]],
            },
            "properties": {
                k: v for k, v in r.items()
                if k not in ("x", "y", "z")
            }
        })
    fc = {"type": "FeatureCollection", "features": features}
    geojson_path.parent.mkdir(parents=True, exist_ok=True)
    with geojson_path.open("w", encoding="utf-8") as f:
        json.dump(fc, f, ensure_ascii=False)
    print(f"GeoJSON written: {geojson_path}  (features={len(features)})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--timeout", type=float, default=5.0)
    parser.add_argument("--step", type=float, default=2.0, help="采样间距（米）")
    parser.add_argument("--out", default="map_waypoints", help="输出文件名前缀（不含扩展名）")
    args = parser.parse_args()

    client = carla.Client(args.host, args.port)
    client.set_timeout(args.timeout)

    world = client.get_world()
    carla_map = world.get_map()
    print(f"Map name: {carla_map.name}")

    rows = list(waypoint_rows_from_map(carla_map, args.step))

    out_prefix = Path(args.out)
    write_csv(rows, out_prefix.with_suffix(".csv"))
    write_geojson(rows, out_prefix.with_suffix(".geojson"))

    # 如果你还想要 OpenDRIVE：
    # xodr = carla_map.to_opendrive()
    # (out_prefix.with_suffix(".xodr")).write_text(xodr, encoding="utf-8")
    # print(f"OpenDRIVE written: {out_prefix.with_suffix('.xodr')}")


if __name__ == "__main__":
    main()
