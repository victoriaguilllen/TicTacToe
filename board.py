
import cv2
from typing import List
import numpy as np
from find_quadrant import *
from scipy.spatial import distance
import time
import copy
import os

current_dir = os.getcwd()

class Board:
    def __init__(self):
        self.grid = self.create_empty_grid() #empty mat 3x3
        self.grid_coor = None #frame coors of the mat 3x3,        
        self.crosses = []
        self.circles = []

        #trajectory
        self.trajectory_only = None

        #imagen en negro solo son el grid
        self.grid_black_backgound = None

    def ocupado(self, position):
        if self.grid[position[0]][position[1]] != 0: 
            return True
        return False

    def create_empty_grid(self):
        grid = [[0, 0, 0] for _ in range(3)]
        return grid
    

    def create_greed_coor(self, frame):
        self.detect_board(frame)


    def add_grid_coor(self, pos, player):
        self.grid[pos[0]][pos[1]] = player
        

    def draw_board(self, frame):
        #draw grid
        if self.grid_coor:
            for y in self.grid_coor['horizontal']:
                cv2.line(frame, (y[0], y[1]), (y[2], y[3]), color=(0, 255, 0), thickness=2)  # Línea horizontal

            for x in self.grid_coor['vertical']:
                cv2.line(frame, (x[0], x[1]), (x[2], x[3]), color=(0, 255, 0), thickness=2)  # Línea vertical

        #draw circles
        for circle in self.circles:
            cv2.circle(frame, circle['center'], circle['figure_width'], color=(0, 0, 255), thickness=2)

        #draw crosses
        for cross in self.crosses:
            cv2.line(frame, cross['line1'][0], cross['line1'][1], color=(0, 0, 255), thickness=2)
            cv2.line(frame, cross['line2'][0], cross['line2'][1], color=(0, 0, 255), thickness=2)

        return frame
    

    def detect_board(self, frame):

        height, width = frame.shape[:2]
        rows, cols = 3, 3

        # Distancia entre cada línea
        row_height = height // rows
        col_width = width // cols

        horizontal_lines = []
        vertical_lines = []

        # Coordenadas de las líneas horizontales (cada una como [x1, y1, x2, y2])
        for i in range(1, rows):
            y = i * row_height
            horizontal_lines.append([0, y, width, y])  # De (0, y) a (width, y)

        # Coordenadas de las líneas verticales (cada una como [x1, y1, x2, y2])
        for i in range(1, cols): 
            x = i * col_width
            vertical_lines.append([x, 0, x, height])  # De (x, 0) a (x, height)

        # Almacenar
        self.grid_coor = {
            'horizontal': horizontal_lines,
            'vertical': vertical_lines
        }

        #imagen negra
        lack_frame = np.zeros_like(frame)

        #crear la imagen en negro
        for y in self.grid_coor['horizontal']:
                cv2.line(lack_frame, (y[0], y[1]), (y[2], y[3]), color=(0, 255, 0), thickness=2)  # Línea horizontal

        for x in self.grid_coor['vertical']:
            cv2.line(lack_frame, (x[0], x[1]), (x[2], x[3]), color=(0, 255, 0), thickness=2)  # Línea vertical

        self.grid_black_backgound = lack_frame


    def save_move(self, player, move):

        #el cuadrante dado por el move se encuentra en la región delimitada por las correspondientes lineas del grid del tablero
        
        #forma del frame
        height = self.grid_coor['vertical'][0][3] 
        width = self.grid_coor['horizontal'][0][2]


        #busco el centro del cuadrante correspondiente
        #1. calculo el centro del primer cuadrante
        center_one = ((width/3)//2, (height/3)//2)

        #2. le sumo la anchura y altura correspondiente en función del cuadrante en el que se encuentre
        center = (int(center_one[0] + move[1]*(width//3)), int(center_one[1]+move[0]*(height//3)))
 

        #dibujo sobre el frame el elemento correspondiente en su posición
        #1. establezo las dimensiones de la figura
        figure_height = int(height//4) 
        figure_width = int(width//4)

        # self.moves.append((player, center, figure_width, figure_height))

        if player == 'O':
            self.circles.append({'center':center, 'figure_width':figure_width//3})

        elif player == 'X':
            self.crosses.append({'line1': [(center[0], center[1] - figure_height // 2), (center[0], center[1] + figure_height // 2)], 'line2': [(center[0] - figure_width // 2, center[1]), (center[0] + figure_width // 2, center[1])]})

   
    def detect_color(self, player, frame):

        #red
        lower_1 = np.array([0, 180, 100])    # Rango para rojos más brillantes
        upper_1 = np.array([10, 255, 255])
        lower_2 = np.array([170, 180, 100])  # Rango para rojos más saturados
        upper_2 = np.array([180, 255, 255])

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask1 = cv2.inRange(hsv, lower_1, upper_1)
        mask2 = cv2.inRange(hsv, lower_2, upper_2)
        mask = cv2.bitwise_or(mask1, mask2)

        mask = cv2.dilate(mask, None, iterations=2)
        mask = cv2.erode(mask, None, iterations=1)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            M = cv2.moments(largest_contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                return True, (cx, cy)
            
        return False, None
    

    def draw_trajectory(self, trajectory, frame):

        self.trajectory_only = trajectory
        if len(trajectory) > 1:
            for i in range(1, len(trajectory)):
                pt1 = trajectory[i - 1]
                pt2 = trajectory[i]
                if distance.euclidean(pt1, pt2) <= 50:
                    cv2.line(frame, pt1, pt2, (0, 255, 255), 2)
        return frame


    def get_trajectory(self, picam):
        # Crear una imagen en blanco (predefinir tamaño del video si no se detecta rojo)
        if not self.trajectory_only:
            return None

        frame_width = picam.preview_configuration.main.size[0]
        frame_height = picam.preview_configuration.main.size[1]
        blank_image = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)

        if len(self.trajectory_only) > 1:
            for i in range(1, len(self.trajectory_only)):
                pt1 = self.trajectory_only[i - 1]
                pt2 = self.trajectory_only[i]
                if distance.euclidean(pt1, pt2) <= 50:
                    # cv2.line(frame, pt1, pt2, (0, 255, 255), 2)
                    cv2.line(blank_image, pt1, pt2, (0, 255, 255), 2)

        #guardar la imgaen 
        cv2.imwrite(f"{current_dir}/new_image/trajectory_only.jpg", blank_image)
        return blank_image
        

