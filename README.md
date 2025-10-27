<h1 style="color:rgb(183, 44, 72);"><strong>TIC-TAC-TOE ❌⭕</strong></h1>

## Adaptation of the popular game “Tic-Tac-Toe” using computer vision. This project was developed as the final assignment for the **Computer Vision** course (3rd year, Mathematical Engineering).

---

<h2 style="color:rgb(238, 137, 137);"><strong>INSTRUCTIONS</strong></h2>

<h3 style="color:rgb(132, 132, 132);"><u>Game Mode Selection</u></h3>
When starting the game, a menu is displayed where you can choose between two modes:
1. **Single Player Mode** – Play against the computer.
2. **Multiplayer Mode** – Play against another player.

<h3 style="color:rgb(132, 132, 132);"><u>Game Start</u></h3>
Once a mode is selected, the game begins after a 3-second countdown to prepare.

<h3 style="color:rgb(132, 132, 132);"><u>Gameplay</u></h3>

- On your turn, place your piece (X or O) in any available cell on the board.  
- If you’re playing against the computer, it will automatically make its move.  
- The player performs movements using a **red-colored object**, which the system detects through **computer vision** to determine where the player is "drawing" in the air.

<h3 style="color:rgb(132, 132, 132);"><u>End of Game Condition</u></h3>
The game ends when one of the players aligns three consecutive symbols horizontally, vertically, or diagonally.  
If the board fills up without a winner, the game ends in a draw.

---

<h2 style="color:rgb(238, 137, 137);"><strong>HOW TO RUN</strong></h2>

Follow these steps to play the game:

1. **Game Initialization:**
   - Run the following file: `game_manager.py`
   - The **bag-of-words model** and **camera calibration** are already pre-trained and configured.

---

<h2 style="color:rgb(238, 137, 137);"><strong>PROJECT STRUCTURE</strong></h2>

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



---

<h2 style="color:rgb(238, 137, 137);"><strong>REQUIREMENTS</strong></h2>

```python
numpy==1.26.0
scikit-learn==1.3.1
pandas==1.5.2
tqdm==4.66.6
opencv-python==4.8.0.76

