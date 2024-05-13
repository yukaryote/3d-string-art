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


def generate_ref_images(fp: str, width: int = 512, height: int = 512, num_views: int = 5):
    """Generate reference images of a mesh as input to optimization. 
    The reference images are taken at uniform angles around the yaw axis of the object.
    Two additional reference images are taken from the top and bottom.
    So the total number of images outputted is num_views + 2.

    Args:
        num_views (int, optional): number of reference images to generate from the yaw axis. Defaults to 5.
    """
    renderers = []
    model = load_mesh(fp)
    for k in range(num_views):
        renderer = o3d.visualization.rendering.OffscreenRenderer(width, height)
        renderer.scene.set_background([1, 1, 1, 1])  # Set background to white
        renderer.scene.add_geometry("model", model)
        renderer.scene.set_lighting(render.scene.LightingProfile.NO_SHADOWS, (0, 0, 0))
        # render at angle 2pi*k/num_views about y-axis
        renderer.scene.camera.look_at(center=[0, 0, 0], eye=[np.cos(2 * np.pi * k / num_views), 0, np.sin(2 * np.pi * k / num_views)], up=[0, 1, 0])
        img_o3d = render.render_to_image()
        o3d.io.write_image("mtest2.jpeg", img_o3d, 9)
        img = np.array(img_o3d)
        cv.imshow("model", img)
        cv.waitKey(1)

def load_frame(fp: str, num_points: int = 500) -> o3d.geometry.LineSet:
    """Load a frame from filepath to connect strings to.

    Args:
        fp (str): file path of mesh, must be .ply. If None, then load Armadillo.

    Returns:
        o3d.geometry.LineSet: LineSet wireframe with just the points.
    """
    if fp is not None:
        mesh = o3d.io.read_triangle_mesh(fp)
    else:
        armadillo_mesh = o3d.data.ArmadilloMesh()
        mesh = o3d.io.read_triangle_mesh(armadillo_mesh.path)
    return mesh


def mesh2wf(mesh: o3d.geometry.TriangleMesh, num_points: int = 500) -> o3d.geometry.LineSet:
    """Convert mesh to wireframe

    Args:
        mesh (o3d.geometry.TriangleMesh): 
        num_points (int, optional): _description_. Defaults to 500.

    Returns:
        o3d.geometry.LineSet: _description_
    """
    pcd = mesh.sample_points_uniformly(number_of_points=num_points)
    wireframe_pcd = o3d.geometry.LineSet()
    # convert point cloud to LineSet with no lines
    wireframe_pcd.points = pcd.points
    wireframe_pcd.lines = o3d.utility.Vector2iVector()
    # create trivial line so that OffscreenRenderer doesn't throw error
    draw_line(wireframe_pcd, 0, 0)
    print("wf points and lines", len(wireframe_pcd.points), len(wireframe_pcd.lines))
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


def extrinsics2lookat(extrinsics: np.array) -> List:
    """Convert 4x4 extrinsics matrix to look_at form becaues OffscreenRenderer doesn't support regular ol' extrinsics matrices??

    Args:
        extrinsics (np.array): shape (4, 4) extrinsics matrix

    Returns:
        List: list of [center, eye, up] vectors
    """
    R = extrinsics[..., :-1, :-1]
    t = extrinsics[..., :-1, -1]
    # eye = 3D global location of the camera
    eye = t
    # center = 3D global location camera is pointed at = w2c_translation + R.inverse @ camera_z-axis
    center = t + np.linalg.inv(R) @ np.array([0., 0., 1.])
    # up = unit vector representing which xyz is the camera's up direction
    up = [0., 1., 0.]
    return [center, eye, up]


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


<<<<<<< HEAD
if __name__ == "__main__":
    # wf = load_frame("data/sphere.ply")

    # for i in range(0, len(wf.points)):
    #     for j in range(i, len(wf.points)):
    #         if np.random.rand() < 0.01:
    #             draw_line(wf, i, j)
    # o3d.visualization.draw_geometries([wf])
    generate_ref_images("data/bun_zipper.ply")
=======
def save_view_point(pcd: o3d.geometry, filename: str, width: int, height: int) -> None:
    """Saves camera params json when user presses 'k' in viz window.

    Args:
        pcd (o3d.geometry): geometry in the scene
        filename (str): .json camera params filename
        width (int): window width
        height (int): window height
    """
    def save_json(vis):
        param = vis.get_view_control().convert_to_pinhole_camera_parameters()
        o3d.io.write_pinhole_camera_parameters(filename, param)
        return False

    key_to_callback = {ord("K"): save_json}

    o3d.visualization.draw_geometries_with_key_callbacks([pcd], key_to_callback, width=512, height=512)


if __name__ == "__main__":
    # wf = load_mesh("data/sphere.ply")
    wf = load_mesh()

    save_view_point(wf, "camera_params/save_view_test_1.json", width=512, height=512)
>>>>>>> e11b1ff1eadbb41cfaea9e71bd19fdadb86ced7c
