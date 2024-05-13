import utils
from typing import List
import open3d as o3d
import cv2 as cv

def optimize(
        images_fp: List[str],
        mesh: str,
        steps: int = 500,
        num_points: int = 500,
        ) -> o3d.geometry.LineSet:
    images = utils.load_images(images_fp)
    wf = utils.load_frame(mesh)
    renderers = []

    # TODO: add interactive stuff here to allow user to select their own camera views. Right now it's a hardcoded json

    for _ in range(len(images)):
        # create new OfflineRenderer for each image
        camera = o3d.io.read_pinhole_camera_parameters("test_params.json")
        renderer = o3d.visualization.rendering.OffscreenRenderer(512, 512)
        renderer.camera = camera
        renderer.scene.add_geometry(wf)
        renderers.append(renderer)

    # render the images
    images = utils.render(renderers=renderers)
    cv.imshow(images[0])


if __name__ == "__main__":
    optimize(images_fp=["data/bunny.png"],
             mesh="data/sphere.ply",
             )