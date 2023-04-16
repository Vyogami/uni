% Generate a random integer between 1 and 50
number = randi([1, 50]);

% Initialize game variables
guesses = 0;
last_guess = NaN;
game_over = false;

% Start game loop
while ~game_over && guesses < 10
    % Get player guess
    guess = input('Guess a number between 1 and 50: ');
    
    % Check if guess is correct
    if guess == number
        fprintf('Congratulations! You won in %d guesses.\n', guesses);
        game_over = true;
    else
        % Check if this is the first guess
        if isnan(last_guess)
            % Check if guess is within 10 of the number
            if abs(guess - number) <= 10
                fprintf('WARM!\n');
            else
                fprintf('COLD!\n');
            end
        else
            % Check if guess is closer than before
            if abs(guess - number) < abs(last_guess - number)
                fprintf('WARMER!\n');
            else
                fprintf('COLDER!\n');
            end
        end
        
        % Update game variables
        guesses = guesses + 1;
        last_guess = guess;
    end
end

% Check if player lost
if guesses >= 10 && ~game_over
    fprintf('Sorry, you lost. The number was %d.\n', number);
end