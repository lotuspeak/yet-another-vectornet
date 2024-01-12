import numpy as np

def normalize_angle(angle: float):
    new_angle = divmod(angle + np.pi, np.pi * 2.0)[1]
    if new_angle < 0.0:
        new_angle = new_angle + np.pi * 2.0 - np.pi
    else:
        new_angle = new_angle - np.pi
    return new_angle

def shift_and_rotate(positons, relative_position, relative_angle):
    """_summary_

    Args:
        positons (_type_): array (n,2)
        relative_position (_type_): (2)
        relative_angle (_type_): _description_
    """
    cosr = np.cos(relative_angle)
    sinr = np.sin(relative_angle)
    rotation_matrix = np.array([[cosr, -sinr],[sinr, cosr]])
    positons_shifted = positons - relative_position
    return np.dot(positons_shifted, rotation_matrix)


def rotate_angle(headings, relative_angle):
    """_summary_
    Args:
        headings (_type_): array []
        return  array
    """
    rotated_headings = headings - relative_angle
    return [normalize_angle(angle) for angle in rotated_headings]