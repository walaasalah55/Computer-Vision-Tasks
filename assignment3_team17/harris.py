from utiles import *


def f_harris(source: np.ndarray, k: float = 0.05 , window_size: int = 3):
    # :param k: Sensitivity factor to separate corners from edges.

    I_x = gradient_x(source, 3)
    I_y = gradient_y(source, 3)

    Ixx = gaussian_filter(I_x**2, 3,3, sigma=1)
    Ixy = gaussian_filter(I_y*I_x, 3,3, sigma=1)
    Iyy = gaussian_filter(I_y**2, 3,3, sigma=1)

    # This is H Matrix
    # [ Ix^2        Ix * Iy ]
    # [ Ix * Iy     Iy^2    ]

    # Harris Response R corner strength function
    # R = det(H) - k(trace(H))^2

    # determinant
    detA = Ixx * Iyy - Ixy ** 2
    # trace
    traceA = Ixx + Iyy
        
    harris_response = detA - k * traceA ** 2

    return harris_response


def categorize_harris_response( img, harris_response, threshold ):

    harris_matrix = np.copy(harris_response)
    max_corner_response = np.max(harris_matrix)

    corner_indices  = np.copy(img)
    edges_indices = np.copy(img)
    
    # We can use these peak values to isolate corners and edges
    # - Corner : R > max_response * threshold
    # - Edge   : R < max_response * threshold
    # - Flat   : R = max_response * threshold

    for rowindex, response in enumerate(harris_matrix ):
        for colindex, r in enumerate(response):
            if r > (max_corner_response * threshold):
                # this is a corner
                corner_indices[rowindex, colindex] = [255,0,0]
            elif r < 0:
                # this is an edge
                edges_indices[rowindex, colindex] = [0,255,0]

    return corner_indices, edges_indices







