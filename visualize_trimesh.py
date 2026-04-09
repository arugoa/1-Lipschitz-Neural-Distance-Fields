import trimesh
import argparse 

#### Commandline ####
parser = argparse.ArgumentParser(
    prog="Training of a 1-Lipschitz architecture",
    description="This scripts runs the training optimization of a 1-Lipschitz neural network on some precomputed point cloud dataset."
)

# dataset parameters
parser.add_argument("surface", type=str, default="output/reconstruction_iso-3.780.obj", help="name of the surface to visualize")
args = parser.parse_args()

# Load or create your mesh
mesh = trimesh.load(args.surface)

# Simple, interactive visualization
mesh.show()