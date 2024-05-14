import utils
import numpy as np
from typing import List
import open3d as o3d
import cv2 as cv
import argparse
import os
import time


def greedy(frame, renderer, material, ref_images, view_params, samples_per_iter, num_iters):
    i = 0
    start_time = time.time()
    best_nail = np.random.choice(len(frame.points), size=1).item()
    nails_left = len(frame.points)
    init_rendered_images = utils.render(renderer, view_params)
    #cv.imshow("rendered string", init_renderered_images[0])
    #cv.waitKey(0)  
    prev_loss = np.mean((init_rendered_images - ref_images)**2)
    print("Initial loss", prev_loss)
    material = o3d.visualization.rendering.MaterialRecord()
    utils.draw_line(frame, 0, 0)
    material.line_width = 0.5
    material.shader = "unlitLine"
    material.base_color = [0, 0, 0, 1]  # Set color to black
    renderer.scene.add_geometry("frame", frame, material)
    rendered_images = utils.render(renderer, view_params)

    while i < num_iters:
        init_rendered_images = utils.render(renderer, view_params)
        #cv.imshow(f"struct at iter {i}", init_rendered_images[0])
        #cv.waitKey(1)   
        prev_loss = np.mean((init_rendered_images - ref_images)**2)

        print(f"Iteration {i} out of {num_iters}, {nails_left} nails left")
        print(f"Loss: {prev_loss}")
        sampled_indices = np.random.choice(len(frame.points), min(samples_per_iter, nails_left), replace=False)
        
        next_best_nail = best_nail
        #print("best nail", best_nail)
        for nail in sampled_indices:
            #print("potential next nail", nail)
            # Simulate the addition of this nail to the frame
            utils.draw_line(frame, best_nail, nail)
            renderer.scene.clear_geometry()
            renderer.scene.add_geometry("frame", frame, material)
            # o3d.visualization.draw_geometries([frame])            
            # Render the frame to get the current image
            rendered_images = utils.render(renderer, view_params)
            #cv.imshow("rendered string", rendered_images[0])
            #cv.waitKey(1)            
            # Calculate the MSE loss with all reference images
            current_loss = np.mean((rendered_images - ref_images) ** 2)
            # Check if this is the best nail so far
            if current_loss < prev_loss:
                print("found improvement")
                print("Loss", current_loss, prev_loss)
                prev_loss = current_loss
                next_best_nail = nail
            
            # Remove the line for the next iteration
            utils.remove_line(frame, best_nail, nail)
        
        # After finding the best nail, add it permanently
        print("found next best nail", best_nail, next_best_nail)
        utils.draw_line(frame, best_nail, next_best_nail)
        renderer.scene.clear_geometry()
        renderer.scene.add_geometry("frame", frame, material)
        nails_left -= 1
        best_nail = next_best_nail
        i += 1

    end_time = time.time()
    duration = end_time - start_time
    minutes = int(duration // 60)
    seconds = int(duration % 60)
    print(f"Algorithm took {minutes} minutes and {seconds} seconds to complete.")
    return frame


def conjugate_gradient(frame, renderer, material, ref_images, view_params, samples_per_iter, num_iters):
    pass


def main(frame_mesh_path: str, reference_path: str, samples_per_iter: int, num_iters: int, use_cg: bool, width: int = 512, height: int = 512):
    print("frame path:", os.path.join(frame_mesh_path))
    frame = utils.load_frame(os.path.join(frame_mesh_path))
    material = o3d.visualization.rendering.MaterialRecord()
    material.line_width = 0.5
    material.shader = "unlitLine"
    material.base_color = [0, 0, 0, 1]  # Set color to black
    ref_images, view_params = utils.generate_ref_images(reference_path, num_views=8)

    # Sample num_points from the frame
    total_points = len(frame.points)
    if samples_per_iter > total_points:
        raise ValueError("Number of points requested exceeds the number of available points in the frame.")

    strings_renderer = utils.create_renderer(frame, width=width, height=width)
    rendered_frame = utils.render(strings_renderer, view_params)
    #frame.lines = o3d.utility.Vector2iVector()
    # cv.imshow("Rendered Frame init", rendered_frame[0])
    # cv.waitKey(0)
    # cv.destroyAllWindows()
    if not use_cg:
        # Placeholder for greedy algorithm function call
        results = greedy(frame, strings_renderer, material, ref_images, view_params, samples_per_iter, num_iters)
    else:
        # Placeholder for conjugate gradient algorithm function call
        results = conjugate_gradient(num_iters)

    # Visualization or further processing can be done here with results
    print("Processing completed.")
    o3d.visualization.draw_geometries([results])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process mesh data.")
    parser.add_argument("--frame_mesh_path", type=str, default="data/sphere.ply", help="Path to the frame mesh file.")
    parser.add_argument("--reference_mesh_path", type=str, default="data/bunny", help="Path to the reference folder that contains the reference mesh.")
    parser.add_argument("--num_iters", type=int, default=500, help="Number of iterations for the algorithm.")
    parser.add_argument("--use_cg", action='store_true', help="Use the conjugate gradient algorithm if set, otherwise use greedy.")

    args = parser.parse_args()
    samples_per_iter = int(args.num_iters / 4)

    main(args.frame_mesh_path, args.reference_mesh_path, samples_per_iter, args.num_iters, args.use_cg)
