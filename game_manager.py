import cv2
from typing import List
from board import Board
from picamera2 import Picamera2
import numpy as np
from computer_player import ComputerPlayer
import time
from bolsa_palabras import *
from find_quadrant import *
import os

from security_system import SecuritySystem
from calibration.calibration import undistort_image


current_dir = os.getcwd()
image_classifier = initialise_classifier()

data = np.load(f'{current_dir}/calibration/calib.npz')
intrinsics = data['intrinsic']
dist_coeffs = data['distortion']

# Para ver el resultado corrigiendo la distorsión calibration = True
calibration = False

class GameManager:
    def __init__(self):

        self.board = Board()
        self.game_mode = None
        self.points_playerX = 0
        self.points_playerO = 0
        self.points_win = 3
        self.exit = False
        self.computer_player = None

        #variables para detectar el moviminto de un jugador
        self.turn = {'trajectory': [], 'last_detected_time': np.inf, 'has_detected_color': False, 'capture_trajectory': False}
        self.timeout_threshold = 4 #segundos máximos  la espera de la detección del mismo color


    def reset_turn(self):
        self.turn['capture_trajectory'] = False
        self.turn['has_detected_color'] = False
        self.turn['trajectory'] = []
        self.turn['last_detected_time'] = np.inf 


#funciones para check la situacion
    def check_winner(self):

        grid = self.board.grid

        #rows
        for row in grid:
            if row[0] == row[1] == row[2] and row[0] != 0:
                return row[0]
            
        #colums
        for col in range(3):
            if grid[0][col] == grid[1][col] == grid[2][col] and grid[0][col] != 0:
                return grid[0][col]

        #diagonals
        if grid[0][0] == grid[1][1] == grid[2][2] and grid[0][0] != 0:
                return grid[0][0]
        
        if grid[0][2] == grid[1][1] == grid[2][0] and grid[0][2] != 0:
            return grid[0][2]
                
        return None
  
    def check_draw(self):
        grid = self.board.grid

        for row in grid:
            if 0 in row:  
                return False

        return True
      

    def check_situation(self):

        if self.check_winner(): 
            return self.check_winner()
        
        elif self.check_draw():
            return 'Draw'
        
        else:
            return True #continue the game


#funciones para el inicio
    def draw_menu(self, frame):

        # Text properties
        font = cv2.FONT_HERSHEY_PLAIN
        font_scale = frame.shape[0] / 400  # Proporcional a la altura del frame
        color = (0, 0, 255)  
        thickness = 2
        line_type = cv2.LINE_AA

        # message
        text_lines = [
            "TIC TAC TOE",
            "Choose a game mode",
            "pressing the corresponding key",
            "s: single                 m: multiplayer"
        ]

        # text size
        text_size_lines = [cv2.getTextSize(text, font, font_scale, thickness)[0] for text in text_lines]
        total_height = sum(size[1] + 1 for size in text_size_lines)  

        # center vertically
        frame_height, frame_width, _ = frame.shape
        start_y = (frame_height - total_height) // 3

        # write
        y = start_y
        for i, text in enumerate(text_lines):
            text_size = text_size_lines[i]
            text_width, text_height = text_size

            # center horizontally
            x = (frame_width - text_width) // 2
            y += text_height + 15  #lines innerspace

            # write
            cv2.putText(frame, text, (x, y), font, font_scale, color, thickness, line_type)

        return frame
    
    def draw_count_down(self, number, frame):

        frame_height, frame_width = frame.shape[:2]

        # Calcular las coordenadas del centro del frame
        center_x, center_y = frame_width // 2, frame_height // 2

        # Definir el texto, la fuente, tamaño, color y grosor
        text = number
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1  # Puedes ajustar el tamaño de la fuente
        color = (0, 255, 0)  # Verde (en formato BGR)
        thickness = 5  # Grosor del texto

        # Obtener el tamaño del texto para centrarlo correctamente
        (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)

        # Calcular las coordenadas para centrar el texto
        text_x = center_x - text_width // 2
        text_y = center_y + text_height // 2

        # Dibujar el texto en el centro
        cv2.putText(frame, text, (text_x, text_y), font, font_scale, color, thickness)

        return frame




#funciones de control del juego
    def play_computer(self):
        #coordinates of the next computer move
        next_move = self.computer_player.computer_move(self.board.grid)

        #se actualizan las coor en el grid
        self.board.add_grid_coor(next_move, 'X')

        #se actualizan en el board
        self.board.save_move('X', next_move)

    def play_player(self, player, frame):

        #variable para indicar si se cambia de turno o no
        wait = True
        
        #saber si ha detectado algún color
        color_detected, coor = self.board.detect_color(player, frame)

        #¿Hay color rojo en el frame acual?
        if color_detected:
            self.turn['has_detected_color'] = True
            self.turn['last_detected_time'] = time.time()
            self.turn['trajectory'].append(coor)
            self.turn['capture_trajectory'] = True

        else: 
            #¿Se ha detectado color en ese mismo turno?
            if self.turn['has_detected_color']:

                #¿Han pasado mas de 4 segundos?
                elapsed_time = time.time() - self.turn['last_detected_time']
                if elapsed_time > self.timeout_threshold:

                    #como ya se habia detectado color, se pupone que ya hay trayectoria, pero hacemos la comprobación igualmente
                    trajectory = self.board.get_trajectory(picam)
                    if trajectory is not None and len(self.turn["trajectory"]) > 30:
                        #analizamos qué es
                        new_data = Dataset.load(f"{current_dir}/new_image", "*.jpg") 
                        shape = predict_new(image_classifier, new_data)

                        #confirmamos que corresponde con el jugador 
                        if (shape[0] == 1 and player != 'X') or (shape[0] == 0 and player != 'O'):
                            self.reset_turn()
                            return wait, frame
                        
                        #buscamos en que cuadrante está
                        next_move = find_quadrant(self.board_points, trajectory)
                        if self.board.ocupado(next_move):
                            self.reset_turn()
                            return wait, frame

                        #se actualizan las coor en el grid
                        self.board.add_grid_coor(next_move, player)

                        #se actualizan en el board
                        self.board.save_move(player, next_move)

                        #pasa al siguiente jugador
                        wait = False 
                    else:
                        self.reset_turn()

                #no han pasado mas de 4 segundos -> solo mostramos la trayectoria hasta ahora
                else:
                    self.turn['capture_trajectory'] = True

            #no se ha detectado color en ese momento, se sigue a la espera
            else:
                self.reset_turn()

        #si se habia detectado o se detecta color, se devuelve el frame con la trayectoria
        if self.turn['capture_trajectory']:
            frame = self.board.draw_trajectory(self.turn["trajectory"], frame)

        return wait, frame



#función principal
    def start_game(self, picam):

        # Variables de tipo juego
        start_mode = False
        security_mode = True 
        game_mode = False
        count_down_mode = False 

        # Variables del juego
        capture_grid = True  # Detectar el grid en el primer frame
        turn = 'X' #comienza siempre el jugador 'X'

        #security mode
        security = SecuritySystem()
        step = 0 #estamos en la primera parte de la contraseña
        end = False
        enter = [0,0,0]

        while security_mode:
            frame = picam.capture_array()
            
            #qué estamos buscando
            color = security.password[step][0]
            shape = security.password[step][1]

            #ha acertado
            if security.detect_color(color, frame) and security.detect_shape(shape, frame):

                if enter[step] == 0:
                    enter[step] = 1
                    step += 1
        
                if 0 not in enter:
                    end = True

            #ha adivinado todo
            if end:
                security_mode = False
                start_mode = True

            frame = security.draw_security_situation(frame, step)
            if calibration:
                frame = undistort_image(frame, intrinsics, dist_coeffs)
            cv2.imshow("Game", frame)

            # Salir al presionar 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break


        # Start mode
        while start_mode:
            frame = picam.capture_array()

            # Salir al presionar 'q'
            key = cv2.waitKey(1) & 0xFF 
            if key == ord('q'):
                break

            # Menú
            frame = self.draw_menu(frame)  # Mostrar el menú
            if calibration:
                frame = undistort_image(frame, intrinsics, dist_coeffs)
            cv2.imshow("Game", frame)

            # Esperar tecla para comenzar
            if key in [ord('s'), ord('m')]:
                if key == ord('s'):  # Contra la computadora, modo individual
                    self.game_mode = 's'
                    self.computer_player = ComputerPlayer()

                elif key == ord('m'):  # Multijugador
                    self.game_mode = 'm'

                start_mode = False
                count_down_mode = True
                time_count = time.time()

        # Count down mode
        number = 3
        while count_down_mode:
            frame = picam.capture_array()

            # Cada 1 segundo cambia el número de la pantalla
            if (time.time() - time_count) > 0.5:
                number -= 1
                time_count = time.time()

                if number == 0:
                    count_down_mode = False  # Se detiene la cuenta atrás
                    game_mode = True
                    cv2.destroyAllWindows()

            # Mostrar el número de la cuenta atrás
            frame = self.draw_count_down(str(number), frame)
            if calibration: 
                frame = undistort_image(frame, intrinsics, dist_coeffs)
            cv2.imshow("Game", frame)

            # Salir al presionar 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break


        #game mode
        self.last_detected_time = time.time() # variable que indica la última vez que se detecto el color rojo
        while game_mode:
            frame = picam.capture_array()

            #solo se hace una vez en toda la partida
            if capture_grid:
                self.board.create_greed_coor(frame)
                self.board_points = detect_corners(self.board.grid_black_backgound)
                capture_grid = False

            #condición de parada, hay o empate o alguien ha ganado
            situation = self.check_situation()
            if situation != True:
                game_mode = False


            #modo de juego contra el ordenador
            if self.game_mode == 's':
                
                #turno del ordenador
                if turn == 'X':
                    self.play_computer()
                    turn = 'O'

                #turno del jugador
                elif turn == 'O':
                    next_player, frame = self.play_player('O', frame)

                    #si se ha identificado patrón, nos cambiamos de turno, sino no
                    if not next_player:
                        turn = 'X'

            #modo de juego multijugador
            elif self.game_mode == 'm':
                next_player, frame = self.play_player(turn, frame)

                #si se ha identificado patrón, nos cambiamos de turno, sino no
                if not next_player:
                    if turn == 'X': 
                        turn = "O"
                    else: 
                        turn = "X"
                    self.reset_turn()

            frame = self.board.draw_board(frame)
            if calibration:
                frame = undistort_image(frame, intrinsics, dist_coeffs)
            cv2.imshow('Game', frame)

            # Salir al presionar 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'): 
                break


        #end game
        situation = self.check_situation()
        if situation == 'Draw': 
            print('Draw')
            picam.stop()
            cv2.destroyAllWindows()
        else:
            if turn == "X":
                print(f'O won.')
            else:
                print("X won.")
            picam.stop()
            cv2.destroyAllWindows()

        picam.stop()
        cv2.destroyAllWindows()

        
       

if __name__ == "__main__":

    picam = Picamera2()
    picam.preview_configuration.main.size = (640, 480)  # Ajusta la resolución según sea necesario
    picam.preview_configuration.main.format = "RGB888"
    picam.preview_configuration.align()
    picam.configure("preview")
    picam.start()

    gm = GameManager()
    gm.start_game(picam)
