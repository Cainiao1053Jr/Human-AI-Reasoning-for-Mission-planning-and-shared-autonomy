import geopandas as gpd
import matplotlib.pyplot as plt

#nodes = gpd.read_file("map2_points.geojson")
#nodes = gpd.read_file("junction_points_with_edges.geojson")
nodes = gpd.read_file("map2_junction_points_with_edges.geojson")
#nodes = gpd.read_file("map2_junction_points.geojson")
#edges = gpd.read_file("nav_edges.geojson")

fig, ax = plt.subplots(figsize=(7, 7))

#edges.plot(ax=ax, linewidth=1, color="black", alpha=0.9)
nodes.plot(ax=ax, markersize=12, color="tab:blue", alpha=0.9)

ax.set_title("CARLA Junction Graph (condensed)")
ax.set_aspect("equal")
ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)")

ax.invert_xaxis()

plt.tight_layout()
plt.show()

# # nodes = gpd.read_file("map_points.geojson")
# nodes = gpd.read_file("junction_points_with_edges.geojson")
# # edges = gpd.read_file("nav_edges.geojson")
#
# fig, ax = plt.subplots(figsize=(7, 7))
#
# # edges.plot(ax=ax, linewidth=1, color="black", alpha=0.9)
# nodes.plot(ax=ax, markersize=12, color="tab:blue", alpha=0.9)
#
# ax.set_title("CARLA Junction Graph (condensed)")
# ax.set_aspect("equal")
# ax.set_xlabel("X (m)")
# ax.set_ylabel("Y (m)")
#
# ax.invert_xaxis()
#
# plt.tight_layout()
# plt.show()