% Create a 3x3 matrix of zeros to represent the game board
board = zeros(3, 3);

% Set the initial player to Player 1 ('X')
current_player = 'X';

% Play the game until there is a winner or the game is a draw
while true
    % Display the current state of the game board
    disp(board);

    % Prompt the current player to make a move
    move = input(sprintf('Player %s, enter your move as [row, column]: ', current_player), 's');
    row = str2double(move(1));
    col = str2double(move(3));

    % Check that the move is valid (i.e., the specified location is empty)
    if board(row, col) ~= 0
        disp('Invalid move. Try again.');
        continue;
    end

    % Update the board with the player's move
    if strcmp(current_player, 'X')
        board(row, col) = 1;
    else
        board(row, col) = -1;
    end

    % Check if the game is over
    if abs(sum(board(:,1))) == 3 || abs(sum(board(:,2))) == 3 || abs(sum(board(:,3))) == 3 ...
            || abs(sum(board(1,:))) == 3 || abs(sum(board(2,:))) == 3 || abs(sum(board(3,:))) == 3 ...
            || abs(sum(diag(board))) == 3 || abs(sum(diag(flip(board)))) == 3
        % The game is over, and the current player has won
        disp(sprintf('Congratulations! Player %s wins!', current_player));
        break;
    elseif sum(abs(board(:))) == 9
        % The game is a draw
        disp('The game is a draw.');
        break;
    else
        % Switch to the other player's turn
        if strcmp(current_player, 'X')
            current_player = 'O';
        else
            current_player = 'X';
        end
    end
end

% Display the final state of the game board
disp(board);

% Draw the game board using plot and text
x = [0 1 2 3; 0 1 2 3; 0 1 2 3; 0 1 2 3];
y = [0 0 0 0; 1 1 1 1; 2 2 2 2; 3 3 3 3];
plot(x, y, 'k', 'LineWidth', 2);
hold on;
plot(x', y', 'k', 'LineWidth', 2);

axis equal off;
set(gca,'YDir','reverse')
for i = 1:3
    for j = 1:3
        if board(i,j) == 1
            text(j-0.5, i-0.5, 'X', 'FontSize', 72, 'HorizontalAlignment', 'center', 'Color', 'r');
        elseif board(i,j) == -1
            text(j-0.5, i-0.5, 'O', 'FontSize', 72, 'HorizontalAlignment', 'center', 'Color', 'g');
        end
    end
end
hold off;
