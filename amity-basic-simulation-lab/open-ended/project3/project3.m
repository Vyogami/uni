% Define the student names and marks
names = {'Ravi', 'Anuj', 'Rashi', 'Danish', 'Ritu'};
english_marks = [77, 45, 75, 65, 76];
physics_marks = [65, 55, 87, 77, 85];
chemistry_marks = [75, 65, 91, 84, 88];
math_marks = [90, 62, 95, 90, 78];
hindi_marks = [84, 70, 80, 55, 65];

% Print the report cards for each student
for i = 1:length(names)
    % Calculate the grades for each subject
    english_grade = mark_to_grade(english_marks(i));
    physics_grade = mark_to_grade(physics_marks(i));
    chemistry_grade = mark_to_grade(chemistry_marks(i));
    math_grade = mark_to_grade(math_marks(i));
    hindi_grade = mark_to_grade(hindi_marks(i));
    
    % Calculate the overall class
    overall_class = calculate_class([english_marks(i), physics_marks(i), chemistry_marks(i), math_marks(i), hindi_marks(i)]);
    
    % Print the report card for the student
    fprintf('Report Card for %s:\n', names{i});
    fprintf('English Grade: %s\n', english_grade);
    fprintf('Physics Grade: %s\n', physics_grade);
    fprintf('Chemistry Grade: %s\n', chemistry_grade);
    fprintf('Math Grade: %s\n', math_grade);
    fprintf('Hindi Grade: %s\n', hindi_grade);
    fprintf('Overall Class: %s\n\n', overall_class);
end

% Define a function to convert marks to grades
function grade = mark_to_grade(mark)
    if mark < 60
        grade = 'D';
    elseif mark < 70
        grade = 'C';
    elseif mark < 85
        grade = 'B';
    else
        grade = 'A';
    end
end

% Define a function to calculate the class average and return the overall class
function class = calculate_class(marks)
    class_average = mean(marks);
    if class_average < 60
        class = 'II';
    elseif class_average < 75
        class = 'I';
    else
        class = 'Distinction';
    end
end
