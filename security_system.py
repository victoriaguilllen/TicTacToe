from typing import List
from board import Board
import cv2
import numpy as np
import time
from bolsa_palabras import *
from find_quadrant import *
from scipy.spatial import distance


class SecuritySystem:
    def __init__(self):
        self.password = {0: ['red', 'square'],  1: ['green', 'circle'], 2: ['blue', 'pentagon']}
        self.threshold = 500 #numero de píxeles mínimo de un color para determinar que esté en la imágen


    def detect_color(self, color, frame):

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        #rangos
        if color == 'red':
            lower_1 = np.array([0, 120, 70])
            upper_1 = np.array([180, 255, 255])
        
        elif color == 'green':
            lower_1 = np.array([35, 50, 50])   
            upper_1 = np.array([85, 255, 255])


        elif color == 'blue':
            lower_1 = np.array([94, 80, 2])
            upper_1 = np.array([126, 255, 255])


        # máscaras
        mask = cv2.inRange(hsv, lower_1, upper_1)

        #númeor de píxeles en la imagen del color indicado
        n_pixels = cv2.countNonZero(mask)

        return n_pixels > self.threshold
    
    
    def detect_shape(self, shape, frame):

        if shape == 'circle':
            sol = self.detect_circle(frame)
        
        elif shape == 'square':
            sol = self.detect_square(frame)

        elif shape == 'pentagon':
            sol =  self.detect_pentagon(frame)

        return sol
    

    def detect_circle(self, frame, dp=1.2, minDist=50, param1=50, param2=30, minRadius=0, maxRadius=0):
        
        #imagen a escala de grises
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        #desenfoque para reducir ruido
        gray = cv2.medianBlur(gray, 5)

        #detectar
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=dp,minDist=minDist, param1=param1, param2=param2, minRadius=minRadius, maxRadius=maxRadius)

        if circles is not None and circles.size > 0:
            return True
        
        return False



    def detect_square(self, frame, area_threshold=1000):
        #imagen a escala de grises
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        #desenfoque para reducir ruido
        blurred = cv2.medianBlur(gray, 5)

        #detectar bordes con canny
        edges = cv2.Canny(blurred, 50, 150)

        #contornos
        contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        #ver si hay cuadrados
        squares = []
        for contour in contours:
            #aproximar el contorno
            epsilon = 0.04 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)

            #ver i tiene 4 vértices yb un áres dentro del threshold
            if len(approx) == 4 and cv2.isContourConvex(approx):
                area = cv2.contourArea(approx)
                if area > area_threshold:
                    # ver si cumple con llas proporciones de un cuadrado -> las mas o menos iguaes
                    (x, y, w, h) = cv2.boundingRect(approx)
                    aspect_ratio = float(w) / h
                    if 0.9 <= aspect_ratio <= 1.1: #relacion de acpecto cercana a 1
                        squares.append(approx)

        if squares != []:
            return True
        
        return False
    


    def detect_pentagon(self, frame, area_threshold=1000):
        #imagen a escala de grises
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        #desenfoque para reducir ruido
        blurred = cv2.medianBlur(gray, 5)

        #detectar bordes con canny
        edges = cv2.Canny(blurred, 50, 150)

        #contornos
        contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        #ver si hay pentágonos
        pentagons = []
        for contour in contours:
            #aproximar el contorno
            epsilon = 0.04 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)

            # ver si el contorno tiene 5 vertices y un ára dentro del threshold
            if len(approx) == 5 and cv2.isContourConvex(approx):
                area = cv2.contourArea(approx)
                if area > area_threshold:
                    pentagons.append(approx)


        if pentagons != []:
            return True
        return False
    

    def draw_security_situation(self, frame, step):

        #dim frame
        height, width, _ = frame.shape

        # params circulos
        radius = 10  
        thickness = -1 
        gap = 15  # espaciado entre círculos
        y_center = 30  #altura 
        x_start = (width - 3 * (2 * radius + gap)) // 2 + radius

        #coordenadas centros
        centers = [
        (x_start, y_center),
        (x_start + 2 * radius + gap, y_center),
        (x_start + 4 * radius + 2 * gap, y_center)]

        #dibuja circuls
        for i, center in enumerate(centers):
            color = (0, 0, 255) if i < (step) else (255, 255, 255)  
            current_thickness = thickness if i < (step) else 2      
            cv2.circle(frame, center, radius, color, current_thickness)

        return frame
    










