# ----------------------------------------------------------------------------
# -                        Open3D: www.open3d.org                            -
# ----------------------------------------------------------------------------
# Copyright (c) 2018-2023 www.open3d.org
# SPDX-License-Identifier: MIT
# ----------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import utils

import open3d as o3d
#conda install pillow matplotlib

if __name__ == "__main__":

    #armadillo_mesh = o3d.data.ArmadilloMesh()
    #mesh = o3d.io.read_triangle_mesh(armadillo_mesh.path)
    mesh = utils.load_mesh("data/bun_zipper.ply")
    mesh.paint_uniform_color([0, 0, 0])
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
    o3d.visualization.draw_geometries([mesh, coordinate_frame])

