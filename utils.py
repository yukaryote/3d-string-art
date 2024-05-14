import os
import open3d as o3d
import numpy as np
from typing import List
import cv2 as cv


def load_images(images_fp: List[str]) -> np.array:
    """Load images from list of files into numpy array, then makes them black and white.
    Returns numpy array of images of shape (num_images, h, w, c)

    Args:
        images_fp (List[str]): list of num_images file paths. Images must be 256 x 256
    Returns:
        np.array: numpy array of images of shape (num_images, h, w)
    """
    imgs = []
    for fp in images_fp:
        img = cv.imread(fp)
        #assert img.shape[:-1] == (256, 256)
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        imgs.append(img)

    return np.array(imgs)


def load_mesh(fp: str):
    """Load the reference mesh to open3d

    Args:
        fp (str): filepath of mesh
    """
    mesh = o3d.io.read_triangle_mesh(fp)
    return mesh


def generate_ref_images(fp: str, width: int = 512, height: int = 512, num_views: int = 5) -> np.array:
    """Generate reference images of a mesh as input to optimization. 
    The reference images are taken at uniform angles around the yaw axis of the object.
    Two additional reference images are taken from the top and bottom.
    So the total number of images outputted is num_views + 2.

    Args:
        fp (str): path to the reference mesh
        width, height (int, optional): width and height of the resulting image
        num_views (int, optional): number of reference images to generate from the yaw axis. Defaults to 5.
    """
    mesh_name = fp.split("/")[-1]
    mesh = load_mesh(os.path.join(fp, f"{mesh_name}.ply"))
    radius = 5.0
    eye = np.array([[radius, 0, radius]], dtype=np.float32).T  # Transposed to [3, 1]
    center = np.array([[0, 0, 0]], dtype=np.float32).T  # Transposed to [3, 1]
    up = np.array([[0, 1, 0]], dtype=np.float32).T  # Transposed to [3, 1]

    # Configure the camera
    renderer = o3d.visualization.rendering.OffscreenRenderer(width, height)
    renderer.scene.set_background([1, 1, 1, 1])  # Set background to white
    material = o3d.visualization.rendering.MaterialRecord()
    material.base_color = [0, 0, 0, 1]  # Set color to black
    renderer.scene.add_geometry("mesh", mesh, material)
    renderer.scene.set_lighting(renderer.scene.LightingProfile.NO_SHADOWS, (0, 0, 0))
    fov = 30.0  # between 5 and 90 degrees
    aspect_ratio = width / height  # azimuth over elevation
    near_plane = 0.1
    far_plane = 50.0
    fov_type = o3d.visualization.rendering.Camera.FovType.Vertical
    renderer.scene.camera.set_projection(fov, aspect_ratio, near_plane, far_plane, fov_type)
    print("created offscreen renderer")

    images = []

    for k in range(num_views):
        # render at angles 2pi * k / num_views around camera
        eye = np.array([[radius * np.cos(2 * np.pi * k / num_views), 0, radius * np.sin(2 * np.pi * k / num_views)]], dtype=np.float32).T
        renderer.scene.camera.look_at(center, eye, up)

        camera_params = o3d.camera.PinholeCameraParameters()
        fx = width / (2 * np.tan(np.deg2rad(fov / 2)))
        fy = fx
        camera_params.intrinsic = o3d.camera.PinholeCameraIntrinsic(
            width=width,  
            height=height,
            fx=fx,
            fy=fy,
            cx=width / 2,  
            cy=height / 2  
        )
        camera_params.extrinsic = np.linalg.inv(renderer.scene.camera.get_model_matrix())
        output_file_path = os.path.join(fp, f"camera_params_view_{k}.json")
        o3d.io.write_pinhole_camera_parameters(output_file_path, camera_params)
        
        img_o3d = renderer.render_to_image()
        o3d.io.write_image(os.path.join(fp, f"ref_view_{k}.png"), img_o3d, 9)
        img = np.array(img_o3d)
        images.append(img)
        cv.imshow("model", img)
        cv.waitKey(1)
    return np.array(images)


def load_frame(fp: str, num_points: int = 500) -> o3d.geometry.LineSet:
    """Load a frame from filepath to connect strings to.

    Args:
        fp (str): file path of mesh, must be .ply
        num_points (int, optional): number of points to sample. Defaults to 500.

    Returns:
        o3d.geometry.LineSet: LineSet wireframe with just the points.
    """
    mesh = o3d.io.read_triangle_mesh(fp)

    # convert mesh to point cloud
    pcd = mesh.sample_points_uniformly(number_of_points=num_points)
    wireframe_pcd = o3d.geometry.LineSet()
    # convert point cloud to LineSet with no lines
    wireframe_pcd.points = pcd.points
    wireframe_pcd.lines = o3d.utility.Vector2iVector()
    return wireframe_pcd


def draw_line(wf: o3d.geometry.LineSet, i: int, j: int) -> None:
    """Draws a line between points i and j in the wireframe.

    Args:
        wf (o3d.geometry.LineSet): LineSet wireframe
        i (int): index of first point
        j (int): index of second point
    """
    assert 0 <= i < len(wf.points) and 0 <= j < len(wf.points)
    wf.lines.append([i, j])


def render(renderers: List[o3d.visualization.rendering.OffscreenRenderer]) -> np.array:
    """Renders the scene from specified cameras.

    Args:
        renderers (List[o3d.visualization.rendering.OffscreenRenderer]): list of renderers to render from

    Returns:
        np.array: resulting camera images of shape (num_images, renderer_h, renderer_w)
    """
    # render LineSet from two cameras
    # TODO: please for the love of god vectorize this

    images = []
    for r in renderers:
        img = np.array(r.render_to_image())
        images.append(img)
    return np.array(images)


if __name__ == "__main__":
    # wf = load_frame("data/sphere.ply")

    # for i in range(0, len(wf.points)):
    #     for j in range(i, len(wf.points)):
    #         if np.random.rand() < 0.01:
    #             draw_line(wf, i, j)
    # o3d.visualization.draw_geometries([wf])
    generate_ref_images("data/bunny")