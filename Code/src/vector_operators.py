import numpy as np


def vec_add(u, v):
    """
    Adds two vectors and normalizes them
    :param u: First vector
    :param v: Second vector
    :return: Returns vector after adding
    """
    result = u + v
    norm = np.linalg.norm(result)
    return result / norm if norm != 0 else result


def vec_sub(u, v):
    """
    Subtracts two vectors and normalizes them
    :param u: First vector
    :param v: Second vector
    :return: Returns vector after subtracting
    """
    result = u - v
    norm = np.linalg.norm(result)
    return result / norm if norm != 0 else result


def vec_mul(u, v):
    """
    Multiplies two vectors and normalizes them
    :param u: First vector
    :param v: Second vector
    :return: Returns vector after multiplying
    """
    result = u * v
    norm = np.linalg.norm(result)
    return result / norm if norm != 0 else result


def vec_prot_div(u, v):
    """
    Divides two vectors and normalizes them (protects against div by 0)
    :param u: First vector
    :param v: Second vector
    :return: Returns vector after dividing
    """
    v = np.where(v == 0, 1, v)  # Ensures den will be 1 if ever 0
    result = u / v
    norm = np.linalg.norm(result)
    return result / norm if norm != 0 else result


def vec_sqr(u):
    """
    Calculates the square of a vector
    :param u: Vector
    :return: Returns vector after squaring
    """
    result = u ** 2
    norm = np.linalg.norm(result)
    return result / norm if norm != 0 else result


def vec_root(u):
    """
    Calculates the root of a vector
    :param u: Vector
    :return: Returns vector after square root
    """
    result = np.sqrt(np.abs(u))
    norm = np.linalg.norm(result)
    return result / norm if norm != 0 else result
