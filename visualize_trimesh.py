import trimesh

# Load or create your mesh
mesh = trimesh.load('output/reconstruction_iso-2.881.obj')

# Simple, interactive visualization
mesh.show()