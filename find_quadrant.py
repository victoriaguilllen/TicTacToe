import numpy as np
import cv2


def shi_tomasi_corner_detection(image: np.array, maxCorners: int, qualityLevel:float, minDistance: int, corner_color: tuple, radius: int):
    '''
    image - Input image
    maxCorners - Maximum number of corners to return. 
                 If there are more corners than are found, the strongest of them is returned. 
                 maxCorners <= 0 implies that no limit on the maximum is set and all detected corners are returned
    qualityLevel - Parameter characterizing the minimal accepted quality of image corners. 
                   The parameter value is multiplied by the best corner quality measure, which is the minimal eigenvalue or the Harris function response. 
                   The corners with the quality measure less than the product are rejected. 
                   For example, if the best corner has the quality measure = 1500, and the qualityLevel=0.01 , then all the corners with the quality measure less than 15 are rejected
    minDistance - Minimum possible Euclidean distance between the returned corners
    corner_color - Desired color to highlight corners in the original image
    radius - Desired radius (pixels) of the circle
    '''
    
    # Input image to Tomasi corner detector should be grayscale 
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply cv2.goodFeaturesToTrack function
    corners = cv2.goodFeaturesToTrack(
        gray, 
        maxCorners=maxCorners, 
        qualityLevel=qualityLevel, 
        minDistance=minDistance
    )

    if corners is not None:
        # Convert corner coordinates to integer values
        corners = np.int0(corners)
        # Return as a flattened array of (x, y) coordinates
        return np.array([corner.ravel() for corner in corners])
    else:
        # If no corners are found, return an empty array
        return np.array([])


def get_quadrant(board_points: np.ndarray, point: np.ndarray) -> str:
    '''
    Determina en qué cuadrante se encuentra un punto respecto a los puntos de intersección del tablero.
    
    Parameters:
        board_points - Array con las coordenadas de las 4 esquinas del tablero [(x1, y1), (x2, y2), (x3, y3), (x4, y4)].
        point - Coordenada del punto a evaluar (x, y).
        
    Returns:
          cuadrante en matriz 3x3  [1,2,3
                                    4,5,6
                                    7,8,9]
    '''
    # Los puntos del tablero son las intersecciones de las dos líneas (líneas horizontales y verticales)
    top_left, top_right, bottom_left, bottom_right = board_points

    y,x = point
    if y < top_left[1]:
        if x < top_left[0]:
            return 1 # Cuadrante izq superior
        elif x < top_right[0]:
            return 2 # Cuadrante medio superior
        else:
            return 3 # Cuadrante derecha superior
    
    elif y < bottom_left[1]:
        if x < top_left[0]:
            return 4 # Cuadrante izq medio
        elif x < bottom_right[0]:
            return 5 # Cuadrante medio medio
        else:
            return 6 # Cuadrante derecha medio
    
    else:
        if x < bottom_left[0]:
            return 7 # Cuadrante izq abajo
        elif x < bottom_right[0]:
            return 8 # Cuadrante medio abajo
        else:
            return 9 # Cuadrante derecha abajo
    

def check_coordinates_in_quadrants(board_points: np.ndarray, coordinates: np.ndarray) -> dict:
    '''
    Verifica en qué cuadrante se encuentran los puntos dados en coordinates respecto a las intersecciones del tablero.
    
    Parameters:
        board_points - Array con las coordenadas de las 4 esquinas del tablero [(x1, y1), (x2, y2), (x3, y3), (x4, y4)].
        coordinates - Lista de coordenadas de los puntos a verificar.
        
    Returns:
        cuadrantes - Un diccionario con la cantidad de puntos en cada cuadrante.
    '''

    cuadrantes = {1: 0, 2: 0, 3: 0, 
                  4: 0, 5: 0, 6: 0, 
                  7: 0, 8: 0, 9: 0}

    for point in coordinates:
        quadrant = get_quadrant(board_points, point)
        cuadrantes[quadrant] += 1
    
    matriz = {1:[0,0], 2:[0,1], 3:[0,2],
              4:[1,0], 5:[1,1], 6:[1,2],
              7:[2,0], 8:[2,1], 9:[2,2]}

    max_quadrant = max(cuadrantes, key=cuadrantes.get)
    return matriz[max_quadrant]



def sort_coordinates(coords):
    """
    Ordena 4 coordenadas dadas de la siguiente forma:
    1: superior izquierdo, 2: superior derecho, 3: inferior izquierdo, 4: inferior derecho
    
    Args:
        coords (list or np.array): Lista o arreglo con las 4 coordenadas en formato (x, y).
        
    Returns:
        list: Lista con las 4 coordenadas ordenadas en el formato (1, 2, 3, 4) especificado.
    """
    top_left = coords[np.argmin(coords[:, 0] + coords[:, 1])]
    top_right = coords[np.argmin(coords[:, 1] - coords[:, 0])]
    bottom_left = coords[np.argmax(coords[:, 1] - coords[:, 0])]
    bottom_right = coords[np.argmax(coords[:, 0] + coords[:, 1])]
    
    return [top_left, top_right, bottom_left, bottom_right]


def detect_corners(grid):
    board_points = shi_tomasi_corner_detection(grid.copy(), maxCorners=4, qualityLevel=0.7, minDistance = 10, corner_color = [0,255,0], radius = 5)
    board_points = sort_coordinates(board_points)
    return board_points


def find_quadrant(board_points, image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    _, binary_image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

    white_pixels = np.where(binary_image == 255)
    coordinates = np.vstack(white_pixels).T

    # Encontrar el cuadrante con más puntos blancos
    cuadrante = check_coordinates_in_quadrants(board_points, coordinates)

    return cuadrante

