import utils
import numpy as np
from typing import List
import open3d as o3d
import cv2 as cv
import argparse
import os
import time


def greedy(ref_images, frame, sampled_nails, num_iters):
    i = 0
    start_time = time.time()
    while i < num_iters:
        ...
    end_time = time.time()
    duration = end_time - start_time
    minutes = int(duration // 60)
    seconds = int(duration % 60)
    print(f"Algorithm took {minutes} minutes and {seconds} seconds to complete.")

def conjugate_gradient(ref_images, frame, sampled_nails, num_iters):
    pass


def visualize(
        images_fp: List[str],
        mesh: str,
        steps: int = 500,
        num_points: int = 500,
        ) -> o3d.geometry.LineSet:
    images = utils.load_images(images_fp)
    wf = utils.load_frame(mesh)
    renderers = []

    # TODO: add interactive stuff here to allow user to select their own camera views. Right now it's a hardcoded json
    #save_view_point(wf, "camera_params/save_view_test_1.json", width=512, height=512)

    for _ in range(len(images)):
        # create new OfflineRenderer for each image
        camera = o3d.io.read_pinhole_camera_parameters("camera_params/armadillo_512.json")
        intrinsics, extrinsics = np.array(camera.intrinsic.intrinsic_matrix), np.array(camera.extrinsic)
        width, height = camera.intrinsic.width, camera.intrinsic.height
        # center, eye, up = extrinsics2lookat(np.array(camera.extrinsic)) LOL THEY DO TAKE IN EXTRINSICS NVM
        renderer = o3d.visualization.rendering.OffscreenRenderer(width, height)
        renderer.setup_camera(intrinsics, extrinsics, width, height)
        #renderer.scene.add_geometry("wf", wf, mtl)
        renderers.append(renderer)

    # render the images
    images = utils.render(renderers=renderers)
    #print(images[0])
    cv.imwrite("data/im.png", images[0])
    print('finished writing')


def main(frame_mesh_path: str, reference_path: str, num_points: int, num_iters: int, use_greedy: bool):
    frame = utils.load_frame(frame_mesh_path)
    ref_images = utils.generate_ref_images(reference_path)

    # Sample num_points from the frame
    total_points = len(frame.points)
    if num_points > total_points:
        raise ValueError("Number of points requested exceeds the number of available points in the frame.")
    
    sampled_indices = np.random.choice(total_points, num_points, replace=False)
    sampled_points = np.array(frame.points)[sampled_indices]

    if use_greedy:
        # Placeholder for greedy algorithm function call
        assert num_iters <= num_points, "Number of iterations must be less than or equal to num_points for greedy"
        results = greedy(ref_images, frame, sampled_points, num_iters)
    else:
        # Placeholder for conjugate gradient algorithm function call
        results = conjugate_gradient(sampled_points, num_iters)

    # Visualization or further processing can be done here with results
    print("Processing completed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process mesh data.")
    parser.add_argument("frame_mesh_path", type=str, default="data/sphere.py", help="Path to the frame mesh file.")
    parser.add_argument("reference_mesh_path", type=str, default="data/bunny", help="Path to the reference mesh file.")
    parser.add_argument("num_points", type=int, help="Number of points to sample from the frame mesh.")
    parser.add_argument("num_iters", type=int, help="Number of iterations for the algorithm.")
    parser.add_argument("--use_greedy", action='store_true', help="Use the greedy algorithm if set, otherwise use conjugate gradient.")

    args = parser.parse_args()

    main(args.frame_mesh_path, args.reference_mesh_path, args.num_points, args.num_iters, args.use_greedy)
