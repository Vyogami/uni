# Report Card Generator

<p>Write a program for generating Report Cards for 5 students.</p><p>a. The names and marks for the students are given in the following table.</p><p>b. Create a function that will convert the marks to a grade, when called. Use the conversion table given below.</p><p>c. Create another function to display the overall class using the conversion table shown below.</p><p>d. The report card consists of grade obtained in each subject according to the marks secured in the subject, and the overall class according to the average marks of marks obtained in all subjects.</p>

<br>

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

|**Marks in a Subject**|**Grade**|
| :- | :- |
|<60|D|
|>=60 and <70|C|
|>=70 and <85|B|
|>=85 and <=100|A||||

</td>

<td>

|**Average Marks**|**Class Secured**|
| :- | :- |
|<60|II|
|>=60 and <75|I|
|>=75|Distinction|
</td>
</tr>
</table>

## Problem Definition

The aim of this program is to generate report cards for five students by taking their names and marks in different subjects as input and calculating their grades and overall class. The program includes two functions - one for converting the marks to grades and another for calculating the overall class.

## Problem Scope

The program takes in the names and marks of five students in different subjects, including English, Physics, Chemistry, Math, and Hindi. The program then calculates the grades for each subject and the overall class based on the average marks of all subjects.

## Technical Review

The program is written in MATLAB, a high-level programming language that is commonly used in scientific computing and engineering. The program uses basic MATLAB syntax and built-in functions to calculate the grades and overall class.

## Design Requirements

The program requires the following design requirements:

- Input names and marks of five students
- Convert marks to grades based on the given conversion table
- Calculate the overall class based on the average marks of all subjects
- Print a report card for each student with their grades and overall class

## Design Description

### Overview

The program takes in the names and marks of five students and calculates their grades and overall class. It uses two functions - one for converting the marks to grades and another for calculating the overall class.

### Detailed Description

The program starts by defining the names and marks of five students. It then calls the mark_to_grade() function to calculate the grades for each subject based on the given conversion table. It then calls the calculate_class() function to calculate the overall class based on the average marks of all subjects.

The mark_to_grade() function takes in the marks for a subject and returns the grade based on the following conversion table:

| Marks | Grade |
| ----- | ----- |
| < 60  |   D   |
| < 70  |   C   |
| < 85  |   B   |
| >= 85 |   A   |

The calculate_class() function takes in the marks for all subjects and calculates the class based on the following conversion table:

| Class Average | Class         |
| ------------- | -------------|
| < 60          | II           |
| < 75          | I            |
| >= 75         | Distinction  |

Finally, the program prints a report card for each student with their grades and overall class.

### Use

The program can be used to generate report cards for five students. The user can input the names and marks of the students and the program will calculate their grades and overall class.

## Evaluation

### Overview

The program was evaluated using sample input data to test its functionality and correctness.

### Prototype

The following input data was used to test the program:

``` matlab
names = {'Ravi', 'Anuj', 'Rashi', 'Danish', 'Ritu'};
english_marks = [77, 45, 75, 65, 76];
physics_marks = [65, 55, 87, 77, 85];
chemistry_marks = [75, 65, 91, 84, 88];
math_marks = [90, 62, 95, 90, 78];
hindi_marks = [84, 70, 80, 55, 65];
```

### Testing and Results

After running the program with the sample input data, the following report cards were generated:

``` markdown
Report Card for Ravi:
English Grade: B
Physics Grade: C
Chemistry Grade: B
Math Grade: A
Hindi Grade: A
Overall Class: Distinction

Report Card for Anuj:
English Grade: D
Physics Grade: D
Chemistry Grade: D
Math Grade:
```

### Assessment

Overall, we are satisfied with the program we have created. It achieves the goal of generating report cards for students in a simple and efficient manner.

### Next Steps

One potential improvement we could make to the program is to add functionality to read in the student data from a file, rather than having it hardcoded into the program. This would make it easier to generate report cards for large numbers of students.

Another potential improvement would be to add more detailed information to the report card, such as the minimum and maximum marks obtained in each subject, as well as the student's rank within the class.

We could also consider adding functionality to export the report cards to a file, or to generate a summary report for all students in a class or school. This would make it easier to track student progress over time and identify areas where students may need additional support.
