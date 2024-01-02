from utils import *
from typing import List
import open3d as o3d
import cv2 as cv

def optimize(
        images_fp: List[str],
        mesh: str = None,
        steps: int = 500,
        num_points: int = 500,
        ) -> o3d.geometry.LineSet:
    images = load_images(images_fp)
    mtl = o3d.visualization.rendering.MaterialRecord()
    mtl.base_color = [1.0, 0.0, 0.0, 1.0]  # RGBA
    mtl.shader = "defaultUnlit"
    wf = load_mesh(mesh)
    renderers = []

    # TODO: add interactive stuff here to allow user to select their own camera views. Right now it's a hardcoded json

    for _ in range(len(images)):
        # create new OfflineRenderer for each image
        camera = o3d.io.read_pinhole_camera_parameters("camera_params/armadillo.json")
        center, eye, up = extrinsics2lookat(np.array(camera.extrinsic))
        renderer = o3d.visualization.rendering.OffscreenRenderer(camera.intrinsic.width, camera.intrinsic.height)
        renderer.scene.camera.look_at(center, eye, up)
        renderer.scene.add_geometry("wf", wf, mtl)
        renderers.append(renderer)

    # render the images
    images = render(renderers=renderers)
    #print(images[0])
    cv.imwrite("data/im.png", images[0])
    print('finished writing')


if __name__ == "__main__":
    optimize(images_fp=["data/bunny.png"],
             mesh=None,
             )
