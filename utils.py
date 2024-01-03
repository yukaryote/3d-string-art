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

def load_mesh(fp: str = None) -> o3d.geometry.TriangleMesh:
    """Load a mesh from filepath.

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


if __name__ == "__main__":
    # wf = load_mesh("data/sphere.ply")
    wf = load_mesh()

    o3d.visualization.draw_geometries([wf], width=512, height=512, left=0, top=0)
