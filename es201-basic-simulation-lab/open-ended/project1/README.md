# Number Guessing Game Project

<p>Write a program for a Number Guessing Game. </p><p>a. Program generates a random integer number between 1 and 50. (Load package pkg load communications, Use randint to generate random number)</p><p>&emsp;b. Player makes a guess. If first guess is </p><p>&emsp;&emsp;i. within 10 of the number, return "WARM!" </p><p>&emsp;&emsp;ii. further than 10 away from the number, return "COLD!" </p><p>&emsp;c. Next Guess onwards, if guess is </p><p>&emsp;&emsp;i. closer than before, return "WARMER!" </p><p>&emsp;&emsp;ii. further away, return "COLDER!" </p><p>&emsp;d. Player gets maximum 10 chances. If the player guesses the number, he wins, else he looses.</p><p></p>

## Problem Definition

### Problem Scope

In this project, we are tasked with creating a number guessing game. The game will generate a random integer number between 1 and 50 and the player will have to guess the number. The game will provide feedback to the player based on their guess, and the player will have a maximum of 10 chances to guess the number.

### Technical Review

To complete this project, we will use the `randi` function from MATLAB to generate a random integer number between 1 and 50. We will also use basic input/output functions to interact with the player and provide feedback based on their guesses.

### Design Requirements

The design for this project must satisfy the following requirements:

- Generate a random integer number between 1 and 50
- Allow the player to make guesses
- Provide feedback to the player based on their guesses
- Limit the player to a maximum of 10 guesses
- Declare the player as the winner if they guess the number correctly within 10 guesses
- Declare the player as the loser if they are unable to guess the number within 10 guesses

## Design Description

### Overview

The number guessing game is a simple game that generates a random integer number between 1 and 50 and asks the player to guess the number. The game provides feedback to the player based on their guesses, allowing them to adjust their guesses accordingly. The game will continue until the player guesses the number correctly or until they run out of guesses.

### Detailed Description

The game starts by generating a random integer number between 1 and 50 using the `randi` function. The player is then prompted to make a guess using the `input` function. If the player's guess is correct, they win the game and the game ends. If the player's guess is incorrect, the game provides feedback based on their guess.

If the player's guess is the first guess, the game checks whether the guess is within 10 of the number. If the guess is within 10 of the number, the game prints "WARM!". If the guess is further than 10 away from the number, the game prints "COLD!".

If the player's guess is not the first guess, the game checks whether the guess is closer to the number than the previous guess. If the guess is closer to the number than the previous guess, the game prints "WARMER!". If the guess is further away from the number than the previous guess, the game prints "COLDER!".

The game allows the player to make a maximum of 10 guesses. If the player is unable to guess the number within 10 guesses, the game declares them as the loser and ends.

### Use

To play the number guessing game, run the code in MATLAB. The game will prompt the player to make a guess and provide feedback based on their guess. The game will continue until the player guesses the number correctly or until they run out of guesses.

## Evaluation

### Overview

The number guessing game was tested using a variety of input values and edge cases to ensure that the game functions correctly.

### Prototype

The game was prototyped using MATLAB. The code for the game was written and tested within the MATLAB environment.

### Testing and Results

The game was tested using a variety of input values and edge cases. The game functioned correctly in all cases, providing feedback to the player based on their guesses and correctly declaring the player as the winner or loser.

### Assessment

The number guessing game satisfies all of the design requirements and functions correctly in a variety of input values and edge cases.

### Next Steps

There are no immediate next steps for this project. The number guessing game is a simple game and the current design satisfies all of
