# Tic-Tac-Toe Game

<p>Write a program for the game of Tic-Tac-Toe.</p><p>a. It is a two-player game. Players play in alternate turns.</p><p>b. Create a 3x3 matrix of zeros for maintaining moves made by the two players. Players enter the location of where they want to place the cross or zero as a vector [row,column].</p><p>c. Moves of player 1 may be maintained as +1 in the matrix, and that of player 2 may be maintained as -1 in the matrix.</p><p>d. If any of the 8 possible outcomes evaluate to +3, player 1 wins. Similarly, if any of the 8 possible outcomes evaluate to -3, player 2 wins. Otherwise, if sum of all absolute value of board entries is 9, that is, if all positions have been occupied, both players loose. </p><p>e. You may use plot and text to draw the game.</p><p></p>

## Problem Definition

This program is a two-player game of Tic-Tac-Toe, where players take turns entering the location of where they want to place their symbol (either "X" or "O") on a 3x3 board. The first player to get three in a row (horizontally, vertically, or diagonally) wins the game. If all positions have been occupied and no player has won, the game is a draw.

### Problem Scope

The program takes input from two players and updates the board with their moves. It then checks if a player has won the game or if the game is a draw. Finally, it displays the final state of the game board and draws it using plot and text.

### Technical Review

The program uses a 3x3 matrix of zeros to represent the game board. The moves of player 1 are maintained as +1 in the matrix, and those of player 2 as -1. The program checks for a win by summing the values of the rows, columns, and diagonals and checking if any of them evaluate to +3 or -3.

### Design Requirements

1. Create a 3x3 matrix of zeros for the game board
2. Prompt players to enter their moves as [row, column]
3. Update the board with the players' moves
4. Check if a player has won the game or if it is a draw
5. Display the final state of the game board
6. Draw the game board using plot and text

## Design description

### Overview

The Tic-Tac-Toe game is implemented using MATLAB programming language. The game is designed as a two-player game, where players take turns to place their marks on a 3x3 game board. The game board is represented by a 3x3 matrix of zeros, with player 1's moves represented as +1 and player 2's moves represented as -1. The game ends when either player wins by achieving a sum of +3 or -3 in any of the 8 possible outcomes, or if all positions on the board are occupied with no winner.

### Detailed description

The program starts by initializing the game board matrix with zeros and the initial player as Player 1 ('X'). The game is played in a while loop until either player wins or the game is a draw. In each iteration of the loop, the current state of the game board is displayed, and the current player is prompted to make a move by specifying the row and column of the cell where they want to place their mark.

The program checks that the move is valid by verifying that the specified location on the board is empty. If the move is valid, the program updates the board with the player's move by setting the corresponding element in the board matrix to +1 or -1, depending on whether the current player is Player 1 or Player 2.

After the player has made their move, the program checks whether the game has ended. If any of the 8 possible outcomes evaluate to +3, Player 1 wins, and if any of the 8 possible outcomes evaluate to -3, Player 2 wins. If all positions on the board are occupied with no winner, the game is a draw.

If the game has not ended, the program switches to the other player's turn by changing the current player from 'X' to 'O' or vice versa.

Once the game has ended, the final state of the game board is displayed, and the program draws the game board using the plot and text functions of MATLAB. The game board is drawn as a 3x3 grid of lines, and the X and O marks are displayed using the text function at the center of the corresponding cell.

### Use

The program can be used to play a game of Tic-Tac-Toe between two players. The game board is displayed on the MATLAB console, and the players make their moves by specifying the row and column of the cell where they want to place their mark. Once the game is over, the final state of the game board is displayed, and the game board is drawn using the plot and text functions of MATLAB.

## Evaluation

### Overview

The program was evaluated based on its ability to correctly implement the rules of Tic-Tac-Toe and provide a user-friendly interface for playing the game. The program was also evaluated based on its ability to draw the game board using the plot and text functions of MATLAB.

### Prototype

The program was implemented and tested in MATLAB R2021a on a Windows 10 machine. The program was tested using a variety of inputs to ensure that it correctly implements the rules of Tic-Tac-Toe and handles invalid moves and end-of-game conditions correctly.

### Testing and results

The program was tested using the following test cases:

1. Valid moves: The program was tested using a series of valid moves to ensure that the program correctly updates the game board and switches between players.

2. Invalid moves: The program was tested using a series of invalid moves to ensure that the program correctly handles invalid moves and prompts the player to make a valid move.

3. Winning conditions: The program was tested using a series of moves that result in a win for Player 1 or Player 2 to ensure that the program correctly identifies the winning

### Assessment

The program meets the design requirements and correctly implements the game of Tic-Tac-Toe. The program is easy to use and provides a command line interface for players to enter their moves.

### Next Steps

Possible next steps for the program include adding a graphical user interface (GUI) to make the game more user-friendly and improving the AI to enable single-player mode. The program could also be extended to support larger game boards or variations of the game.
