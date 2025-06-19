#
# 3D Affine transformation
#

import math
import random

import numpy as np
from scipy import ndimage
from skimage import transform


def matrix_Rx(angle):
    """
    Calculates matrix for rotation around x axis.
    # Tait-Bryan angles in homogeneous form,
    reference: https://people.cs.clemson.edu/~dhouse/courses/401/notes/affines-matrices.pdf
    Argument:
        angle (float): The angle of rotation in degrees.
    Return:
        matrix (array): matrix of scipy.ndimage.affine_transform
    """
    theta = np.deg2rad(angle)
    Rx = [[math.cos(theta), math.sin(theta), 0], [-math.sin(theta), math.cos(theta), 0], [0, 0, 1]]
    return np.array(Rx)


def matrix_Ry(angle):
    """
    Calculates matrix for rotation around y axis.
    # Tait-Bryan angles in homogeneous form,
    reference: https://people.cs.clemson.edu/~dhouse/courses/401/notes/affines-matrices.pdf
    Argument:
        angle (float): The angle of rotation in degrees.
    Return:
        matrix (array): matrix of scipy.ndimage.affine_transform
    """
    theta = np.deg2rad(angle)
    Ry = [[math.cos(theta), 0, -math.sin(theta)], [0, 1, 0], [math.sin(theta), 0, math.cos(theta)]]
    return np.array(Ry)


def matrix_Rz(angle):
    """
    Calculates matrix for rotation around z axis.
    # Tait-Bryan angles in homogeneous form,
    reference: https://people.cs.clemson.edu/~dhouse/courses/401/notes/affines-matrices.pdf
    Argument:
        angle (float): The angle of rotation in degrees.
    Return:
        matrix (array): matrix of scipy.ndimage.affine_transform
    """
    theta = np.deg2rad(angle)

    Rz = [[1, 0, 0], [0, math.cos(theta), math.sin(theta)], [0, -math.sin(theta), math.cos(theta)]]
    return np.array(Rz)


def rotate_image_center(image, angle, axis):
    """
    Rotates an image around a specified axis.

    Args:
        image (arr): The input image.
        angle (float): The angle of rotation in degrees.
        axis (str): The axis of rotation ("x" or "y").

    Returns:
        The rotated image.
    """

    # Get rotation matrix based on axis
    if axis == "x":
        rotation_matrix = matrix_Rx(angle)
    elif axis == "y":
        rotation_matrix = matrix_Ry(angle)
    else:
        raise ValueError("Invalid axis. Please use 'x' or 'y'.")

    # Calculate offset for centering rotation
    offset = np.array(image.shape) - rotation_matrix.dot(np.array(image.shape))
    offset = offset / 2.0

    # Apply affine transformation
    out_image = ndimage.affine_transform(
        image, rotation_matrix, output_shape=image.shape, offset=offset
    )

    return out_image


def generate_randomtf(n_transforms=100, sx_range=(1.0, 1.3), sy_range=(1.0, 1.1)):
    """
    Generates random scaling factors for image augmentation: slice should be stretched >1.

    Args:
        n_transforms (int): Number of transformations to generate.
        sx_range (float): Range of x-axis scaling factors.
        sy_range (float): Range of y-axis scaling factors.

    Returns:
        A tuple of lists: (sx_list, sy_list)
    """
    sxlist, sylist = [], []
    for _ in range(n_transforms):
        sx = random.uniform(*sx_range)
        sy = random.uniform(*sy_range)
        sxlist.append(sx)
        sylist.append(sy)
    return sxlist, sylist


def stretch_image(image, sx, sy, shearx_angle, sheary_angle):
    """
    Shears an image using the map_coordinate function.

    Args:
    image (arr): The input image.
    sx (float): Scaling x.
    sy (float): Scaling y.
    shearx_angle (float): Shear angle in degrees.
    sheary_angle (float): Shear angle in degrees.

    Returns:
    The sheared image as a NumPy array.
    """

    image_height, image_width = image.shape[:2]

    new_x1, new_y1 = ndimage.map_coordinate(
        image_width, image_height, sx, sy, shearx_angle, sheary_angle
    )
    new_x2, new_y2 = ndimage.map_coordinate(0, 0, sx, sy, shearx_angle, sheary_angle)
    new_x3, new_y3 = ndimage.map_coordinate(image_width, 0, sx, sy, shearx_angle, sheary_angle)
    new_x4, new_y4 = ndimage.map_coordinate(0, image_height, sx, sy, shearx_angle, sheary_angle)

    output_image = np.zeros(
        (
            max(int(new_y1), int(new_y2), int(new_y3), int(new_y4), image_height) + 1,
            max(int(new_x1), int(new_x2), int(new_x3), int(new_x4), image_width) + 1,
        ),
        dtype=np.uint8,
    )

    for y in range(image_height):
        for x in range(image_width):
            new_x, new_y = ndimage.map_coordinate(x, y, sx, sy, shearx_angle, sheary_angle)
            output_image[int(new_y), int(new_x)] = image[y, x]

    return output_image


def rescale_image(image, sx, sy):
    """
    Rescales an image.

    Args:
        image (arr): The input image.
        sx (float): the scaling factor for the x axes.
        sy (float): the scaling factor for the y axes.

    Returns:
        The rescaled image.
    """

    # sy, sx = scale
    augmented_image = transform.rescale(
        image, scale=(sy, sx), preserve_range=True, clip=True
    ).astype(np.uint8)
    return augmented_image


def apply_randomtf(image, sxlist, sylist):
    """
    Applies random scaling transformations to an image.

    Args:
        image (arr): The input image.
        sxlist (list): A list of x-axis scaling factors.
        sylist (list): A list of y-axis scaling factors.

    Returns:
        A list of augmented images.
    """

    augmented_images = []
    for sx, sy in zip(sxlist, sylist):
        if image.ndim == 2:
            augmented_image = rescale_image(image, sx, sy)
        elif image.ndim == 3:
            augmented_image = []
            for iz in range(image.shape[0]):
                img = image[iz]
                augmented_image.append(rescale_image(img, sx, sy))
            augmented_image = np.array(augmented_image, dtype=np.uint8)

        augmented_images.append(augmented_image)
    return augmented_images


def pad_image(image):
    """
    Makes an image squared with padding.

    Args:
        image (arr): The input image.

    Returns:
        The padded image.
    """

    def pad_image_2d(image):
        max_size = max(image.shape)
        padded_image = np.zeros((max_size, max_size), dtype=np.uint8)
        height, width = image.shape
        height_margin, width_margin = (max_size - height) // 2, (max_size - width) // 2
        if height > width:
            padded_image[
                0:max_size,
                width_margin : width_margin + width,
            ] = image
        else:
            padded_image[
                height_margin : height_margin + height,
                0:max_size,
            ] = image
        return padded_image

    if image.ndim == 2:
        padded_image = pad_image_2d(image)
    elif image.ndim == 3:
        max_size = max(image.shape)
        padded_image = np.zeros((image.shape[0], max_size, max_size), dtype=np.uint8)
        for iz in range(image.shape[0]):
            padded_image[iz] = pad_image_2d(image[iz])

    return padded_image


def canvas_size(image, image_size):
    """
    Makes a squared image of image_size with padding.

    Args:
        image (arr): The input image.
        image_size (int): The desired size of the canvas.

    Returns:
        The padded image centered on a canvas of the specified size.
    """

    padded_image = pad_image(image)
    canvas_image = np.zeros((image_size, image_size), dtype=np.uint8)
    margin = (image_size - padded_image.shape[0]) // 2

    canvas_image[margin : margin + image.shape[0], margin : margin + image.shape[1]] = padded_image
    return canvas_image
