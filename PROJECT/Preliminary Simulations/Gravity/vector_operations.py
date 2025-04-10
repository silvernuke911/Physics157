import numpy as np
def sign(x):
    return int(abs(x)/x)
def mag(vector):
    return np.linalg.norm(vector)
def normalize(vector):
    v_mag=mag(vector)
    if v_mag == 0:
        raise ValueError('Cannot normalize a zero vector')
    return vector/ v_mag
def vdot(vector1,vector2):
    return np.dot(vector1,vector2)
def vcrs(vector1,vector2):
    return np.cross(vector1,vector2)
def vprj(vector1,vector2):
    mag_v2 = np.linalg.norm(vector2)
    if mag_v2 == 0:
        raise ValueError('Magnitude of divisor is 0')
    k = np.dot(vector1, vector2) / (mag_v2**2)
    return k * vector2
def vxcl(vector1,vector2):
    output=vector1-vprj(vector1,vector2)
    return output
def vang(vector1, vector2, deg=True):
    mag1 = mag(vector1)
    mag2 = mag(vector2)
    if mag1 == 0 or mag2 == 0:
        raise ValueError('vector magnitude is zero')
    dot_product = vdot(vector1, vector2)
    cos_angle = dot_product / (mag1 * mag2)
    # Clamping the cosine value to the range [-1, 1]
    cos_angle = np.clip(cos_angle, -1, 1)
    angle_rad = np.arccos(cos_angle)
    if deg:
        return np.degrees(angle_rad)
    return angle_rad
def gram_schmidt(*vectors):
    # Check for zero magnitude vectors
    for vector in vectors:
        if np.linalg.norm(vector) == 0:
            raise ValueError('Vector magnitude is zero')
    # Initialize the list of orthogonalized vectors
    orthogonal_vectors = []
    for vector in vectors:
        orthogonal_vector = vector.astype(float).copy()  # Convert to float dtype
        for orth_vec in orthogonal_vectors:
            orthogonal_vector = orthogonal_vector - vprj(orthogonal_vector, orth_vec).astype(float)
        
        orthogonal_vector = normalize(orthogonal_vector)
        orthogonal_vectors.append(orthogonal_vector)
    return np.array(orthogonal_vectors)
def vperp(vector):
    if vector.shape != (2,):
        raise ValueError("Input vector must be a 2D vector (shape (2,)).")
    # 90 degrees counterclockwise rotation
    perp_ccw = np.array([-vector[1], vector[0]])
    # 90 degrees clockwise rotation
    perp_cw = np.array([vector[1], -vector[0]])
    return perp_ccw, perp_cw