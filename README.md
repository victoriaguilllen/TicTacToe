<h1 style="color:rgb(183, 44, 72);"><strong>TIC-TAC-TOE ❌⭕</strong></h1>

## Adaptación del popular juego 'Tres en raya' usando la visión por ordenador. Se trata del proyecto final de la asignatura de Visión por Ordenador de 3º de IMat. 


<h2 style="color:rgb(238, 137, 137);"><strong>INSTRUCCIONES</strong></h2>

<h3 style="color:rgb(132, 132, 132);"><u>Selección modo de juego</u></h3>
Al iniciar el juego, se presenta un menú donde puedes elegir entre dos modos:
1. **Modo Individual**: Juegas contra la computadora.
2. **Modo Multijugador**: Juegas contra otro jugador.

<h3 style="color:rgb(132, 132, 132);"><u>Inicio del Juego</u></h3>
Una vez seleccionado el modo, el juego comenzará con una cuenta regresiva de 3 segundos para prepararse.

<h3 style="color:rgb(132, 132, 132);"><u>Desarrollo del juego</u></h3>

- En tu turno, realiza tu movimiento colocando tu pieza (X o O) en el cuadrante del tablero que elijas.
- Si estás jugando contra la computadora, el ordenador hará su movimiento automáticamente.
- El jugador debe colocar la pieza en un espacio disponible, que será detectado por el sistema, el cual usa la tecnología de visión por computadora para identificar la ubicación donde el jugador está "dibujando" en el aire con el color rojo.

<h3 style="color:rgb(132, 132, 132);"><u>Condición fin de juego</u></h3>
El juego termina cuando uno de los jugadores logra alinear tres símbolos consecutivos en fila, columna o diagonal. Si el tablero se llena sin que nadie haya ganado, el juego termina en un empate.


<h2 style="color:rgb(238, 137, 137);"><strong>PASOS A REPRODUCIR</strong></h2>
Sigue los siguientes pasos para jugar al juego:

   
1. **Inicialización del juego:**
   - Correr el siguiente archivo : `game_manager.py`
   - El entrenamineto de la bolsa de palabras y la calibración ya se ha hecho.




<h2 style="color:rgb(238, 137, 137);"><strong>ESTRUCTURA</strong></h2>

```
|-- Final Project
|   |-- Calibration
|   |   |-- calibration.py
|   |   |-- save_photos.py
|   |-- word_bag
|   |   |-- word_bag.py
|   |   |-- frame_detection.py
|   |   |-- bow.py
|   |   |-- dataset.py
|   |   |-- image_classifier.py
|   |   |-- results.py
|   |   |-- utils.py
|   |-- game_manager.py
|   |-- security_system.py
|   |-- find_quadrant.py
|   |-- computer_player.py
|   |-- board.py
```


<h2 style="color:rgb(238, 137, 137);"><strong>REQUIREMENTS</strong></h2>

```python
numpy==1.26.0
scikit-learn==  1.3.1
pandas==1.5.2
tqdm==4.66.6
opencv-python==4.8.0.76
```


<h2 style="color:rgb(238, 137, 137);"><strong>RECOMENDACIONES</strong></h2>
