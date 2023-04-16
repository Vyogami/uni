# Basic Simulation Lab - Open-Ended Experiments

This directory contains the code and reports for the open-ended Experiments in the Basic Simulation Lab course at Amity University.

## Structure

The directory is organized into separate subdirectories for each experiment. Each subdirectory contains:

- `experimentX.m`: the MATLAB code for the experiment
- `experimentX-observations.pdf`: a PDF file containing the results of running the code
- `README.md`: a report summarizing the approach and results for the experiment

### Example

``` yaml
experiment1/
├── experiment1.m
├── experiment1-observations.pdf
└── README.md
```

## Experiments

|Sr. No.|Aim |
| :- | :- |
|1\. |<p>Write a program for a Number Guessing Game. </p><p>a. Program generates a random integer number between 1 and 50. (Load package pkg load communications, Use randint to generate random number)</p><p>&emsp;b. Player makes a guess. If first guess is </p><p>&emsp;&emsp;i. within 10 of the number, return "WARM!" </p><p>&emsp;&emsp;ii. further than 10 away from the number, return "COLD!" </p><p>&emsp;c. Next Guess onwards, if guess is </p><p>&emsp;&emsp;i. closer than before, return "WARMER!" </p><p>&emsp;&emsp;ii. further away, return "COLDER!" </p><p>&emsp;d. Player gets maximum 10 chances. If the player guesses the number, he wins, else he looses.</p><p></p>|
|2\.|<p>Write a program for a Dice Rolling Game. </p><p>a. Simulate the game as a mutli-player game. Ask the user how many players will play the game. (at least two players must play the game.). </p><p>&emsp;b. Players play in alternate turns. In one round, each player gets a chance to roll the dice.</p><p>&emsp;c. For each turn, the program generates two random integer number between 1 and 6 to simulate rolling of two dices. Display number on dice for each player. (Load package pkg load communications, Use randint to generate random number)</p><p>&emsp;d. When each player has rolled the dice, the player with the highest number wins that round. Display winner of each round.</p><p>&emsp;e. Game goes on for 5 rounds.</p><p>&emsp;f. Overall winner(s) is the player who wins most number of rounds.</p><p>Create appropriate displays to display winner of each round and the overall winner.</p>|
|3\.|<p>Write a program for generating Report Cards for 5 students.</p><p>a. The names and marks for the students are given in the following table.</p><p>b. Create a function that will convert the marks to a grade, when called. Use the conversion table given below.</p><p>c. Create another function to display the overall class using the conversion table shown below.</p><p>d. The report card consists of grade obtained in each subject according to the marks secured in the subject, and the overall class according to the average marks of marks obtained in all subjects.</p>|

### Reference for Experiment 3
<table>
<tr><td>

||English|Physics|Chemistry|Math|Hindi|
| :- | :- | :- | :- | :- | :- |
|Ravi|77|65|75|90|84|
|Anuj|45|55|65|62|70|
|Rashi|75|87|91|95|80|
|Danish|65|77|84|90|55|
|Ritu|76|85|88|78|65|

</td><td>

|**Marks in a Subject**|**Grade**||**Average Marks**|**Class Secured**|
| :- | :- | :- | :- | :- |
|<60|D||<60|II|
|>=60 and <70|C||>=60 and <75|I|
|>=70 and <85|B||>=75|Distinction|
|>=85 and <=100|A||||

</td>
</tr>
</table>

|Sr| Aim|
| :- | - |
|4\. |<p>Write a program for the game of Tic-Tac-Toe.</p><p>a. It is a two-player game. Players play in alternate turns.</p><p>b. Create a 3x3 matrix of zeros for maintaining moves made by the two players. Players enter the location of where they want to place the cross or zero as a vector [row,column].</p><p>c. Moves of player 1 may be maintained as +1 in the matrix, and that of player 2 may be maintained as -1 in the matrix.</p><p>d. If any of the 8 possible outcomes evaluate to +3, player 1 wins. Similarly, if any of the 8 possible outcomes evaluate to -3, player 2 wins. Otherwise, if sum of all absolute value of board entries is 9, that is, if all positions have been occupied, both players loose. </p><p>e. You may use plot and text to draw the game.</p><p></p>|
|5\. |<p>Read a gray scale image in Matlab/Octave. </p><p>(Use cameraman.tif from https://people.math.sc.edu/Burkardt/data/tif/tif.html, and available in Files Tab of this team)</p><p>Obtain a negative of a gray scale image.  Plot the original image and its negative.</p><p>Also plot the image after brightening and darkening it. Use subplot.</p><p></p>|
|6\. |<p>Read a gray scale image in Matlab/Octave. </p><p>(Use screws.tif from https://people.math.sc.edu/Burkardt/data/tif/tif.html, and available in Files Tab of this team)</p><p>a) Plot its histogram and determine a suitable value for threshold.  </p><p>b) Display the original and thresholded image.</p><p></p>|
|7\. |<p>Read a gray scale image in Matlab/Octave. </p><p>(Use circuits.tif available in Files Tab of this team)</p><p>a) Add noise (Gaussian, Salt and Pepper noise) to the image.  </p><p>b) Use filters (Gaussian, Median filters) to reduce noise in the image.</p><p>c) Display the original, noisy and smoothened images.</p><p></p>|
|8\.|<p>Read a color (RGB) image in Matlab/Octave. </p><p>(Use Nemo.PNG available in Files Tab of this team)</p><p>a) Plot the red, green and blue channel on separate plots. Use subplot.</p><p>b) Select a suitable channel for thresholding the foreground. Plot the histogram of this channel.</p><p>c) Use thresholding to convert the image to black and white image and plot it.</p><p></p>|
|9\.|Given the data about Blood Pressure and Cholesterol for 20 patients, using k-means clustering to group the patients into having high risk of heart attack and those having low risk of heart attack.|

### Reference for Experiment 9
|Column 1|Column 2|
| :- | :- |
|Blood Pressure|Cholesterol|

## References

The `references/` directory contains reference materials for the course, including the open-ended report format, open-ended Experiments in docx format etc

For further information, please refer to the individual `README.md` files located in the subdirectories for each experiment.
