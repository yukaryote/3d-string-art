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


def create_renderer(mesh, width: int = 512, height: int = 512, fov: int = 30.0,) -> o3d.visualization.rendering.OffscreenRenderer:
    """Creates an OffscreenRenderer

    Args:
        mesh: _description_
        width (int, optional): _description_. Defaults to 512.
        height (int, optional): _description_. Defaults to 512.
        fov (int, optional): _description_. Defaults to 30.0.

    Returns:
        o3d.visualization.rendering.OffscreenRenderer: renderer
    """
    # mesh = load_mesh(os.path.join(mesh_path))
    renderer = o3d.visualization.rendering.OffscreenRenderer(width, height)
    renderer.scene.set_background([1, 1, 1, 1])  # Set background to white
    material = o3d.visualization.rendering.MaterialRecord()
    if isinstance(mesh, o3d.geometry.LineSet):
        print("Num points", len(mesh.points))
        draw_line(mesh, 0, 0)
        material.line_width = 0.5
        material.shader = "unlitLine"
    else:
        mesh.compute_vertex_normals()
        material.shader = "defaultUnlit"
    material.base_color = [0, 0, 0, 1]  # Set color to black
    renderer.scene.clear_geometry()
    renderer.scene.add_geometry("mesh", mesh, material)
    #renderer.scene.set_lighting(renderer.scene.LightingProfile.NO_SHADOWS, (0, 0, 0))
    renderer.scene.set_lighting(renderer.scene.LightingProfile.MED_SHADOWS, (0.577, -0.577, -0.577))

    aspect_ratio = width / height  # azimuth over elevation
    near_plane = 0.1
    far_plane = 50.0
    fov_type = o3d.visualization.rendering.Camera.FovType.Vertical
    renderer.scene.camera.set_projection(fov, aspect_ratio, near_plane, far_plane, fov_type)

    print("created offscreen renderer")
    return renderer


def generate_ref_images(mesh_path: str, width: int = 512, height: int = 512, num_views: int = 5) -> np.array:
    """Generate reference images of a mesh as input to optimization. 
    The reference images are taken at uniform angles around the yaw axis of the object.

    Args:
        fp (str): path to the reference mesh
        width, height (int, optional): width and height of the resulting image
        num_views (int, optional): number of reference images to generate from the yaw axis. Defaults to 5.
    """
    fov = 30.0  # between 5 and 90 degrees
    mesh_name = mesh_path.split("/")[-1]
    mesh = load_mesh(os.path.join(mesh_path, f"{mesh_name}.ply"))
    renderer = create_renderer(mesh, width=width, height=height, fov=fov)


    radius = 5.0
    eye = np.array([[radius, 0, radius]], dtype=np.float32).T  # Transposed to [3, 1]
    center = np.array([[0, 0, 0]], dtype=np.float32).T  # Transposed to [3, 1]
    up = np.array([[0, 1, 0]], dtype=np.float32).T  # Transposed to [3, 1]
    images = []
    view_params = []

    for k in range(num_views):
        # render at angles 2pi * k / num_views around camera
        eye = np.array([[radius * np.cos(2 * np.pi * k / num_views), 0, radius * np.sin(2 * np.pi * k / num_views)]], dtype=np.float32).T
        renderer.scene.camera.look_at(center, eye, up)

        view_params.append((center, eye, up))

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
        output_file_path = os.path.join(mesh_path, f"camera_params_view_{k}.json")
        o3d.io.write_pinhole_camera_parameters(output_file_path, camera_params)

        img_o3d = renderer.render_to_image()
        o3d.io.write_image(os.path.join(mesh_path, f"ref_view_{k}.png"), img_o3d, 9)
        img = np.array(img_o3d)
        img = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
        images.append(img)
        #cv.imshow("model", img)
        #cv.waitKey(1)
    return np.array(images), view_params


def load_frame(fp: str, num_points: int = 250) -> o3d.geometry.LineSet:
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
    wireframe_pcd.points = o3d.utility.Vector3dVector(np.asarray(pcd.points))
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


def remove_line(wf: o3d.geometry.LineSet, i: int, j: int) -> None:
    """Removes the line between points i and j, if it exists

    Args:
        wf (o3d.geometry.LineSet): LineSet wireframe
        i (int): index of first point
        j (int): index of second point
    """
    line = [i, j]
    if line in wf.lines:
        wf.lines.remove(line)
    else:
        line = [j, i]
        if line in wf.lines:
            wf.lines.remove(line)


def render(renderer: o3d.visualization.rendering.OffscreenRenderer, view_params: List) -> np.array:
    """Renders the scene from specified cameras.

    Args:
        renderers (List[o3d.visualization.rendering.OffscreenRenderer]): list of renderers to render from

    Returns:
        np.array: resulting camera images of shape (num_images, renderer_h, renderer_w)
    """
    # render LineSet from two cameras
    # TODO: please for the love of god vectorize this

    images = []
    for param in view_params:
        center, eye, up = param
        renderer.scene.camera.look_at(center, eye, up)
        img_o3d = renderer.render_to_image()
        img = np.array(img_o3d)
        img = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
        images.append(img)
        # cv.imshow("render", img)
        # cv.waitKey(100)
    return np.array(images)


def dummy(frame, renderer):
    material = o3d.visualization.rendering.MaterialRecord()
    if isinstance(frame, o3d.geometry.LineSet):
        print("Num points", len(frame.points))
        draw_line(frame, 0, 0)
        material.line_width = 0.5
        material.shader = "unlitLine"
    material.base_color = [0, 0, 0, 1]  # Set color to black
    renderer.scene.clear_geometry()  # Clear previous geometries
    renderer.scene.add_geometry("mesh", frame, material)
    for i in range(0, len(frame.points)):
        for j in range(i, len(frame.points)):
            if np.random.rand() < 0.01:
                draw_line(frame, i, j)
    renderer.scene.add_geometry("modified_frame", frame, material)
    rendered_images = render(renderer, view_params)
    cv.imshow("TEST IN GREEDY", rendered_images[0])
    cv.waitKey(1000)


if __name__ == "__main__":
    wf = load_frame("data/sphere.ply")

    for i in range(0, len(wf.points)):
        for j in range(i, len(wf.points)):
            if np.random.rand() < 0.01:
                draw_line(wf, i, j)
    # o3d.visualization.draw_geometries([wf])
    images, view_params = generate_ref_images("data/sphere", num_views=5)
    renderer = create_renderer(wf)
    eye = np.array([[5, 0, 5]], dtype=np.float32).T  # Transposed to [3, 1]
    center = np.array([[0, 0, 0]], dtype=np.float32).T  # Transposed to [3, 1]
    up = np.array([[0, 1, 0]], dtype=np.float32).T  # Transposed to [3, 1]
    images = render(renderer, view_params=view_params)
    cv.imshow("TEST OUTSIDE GREEDY", images[0])
    cv.waitKey(1000)

    wf2 = load_frame("data/sphere.ply")
    renderer2 = create_renderer(wf2)
    dummy(frame=wf2, renderer=renderer2)

    