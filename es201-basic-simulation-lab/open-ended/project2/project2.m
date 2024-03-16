num_players = input('Enter number of players (at least two players): ');

% Initialize player scores
scores = zeros(1, num_players);

% Initialize overall max score to be zero
overall_scores = zeros(1, num_players);

% Game goes on for 5 rounds
for round = 1:5
    fprintf('\nRound %d:\n', round);
    
    % Players play in alternate turns
    for player = 1:num_players
        fprintf('\nPlayer %d turn:\n', player);
        
        % Generate two random integer number between 1 and 6 to simulate rolling of two dices
        dice1 = randi([1,6]);
        dice2 = randi([1,6]);
        
        % Display number on dice for each player
        fprintf('Dice 1: %d\n', dice1);
        fprintf('Dice 2: %d\n', dice2);
        
        % Calculate total score for this turn
        score = dice1 + dice2;
        
        % Update player score
        scores(player) = scores(player) + score;
        overall_scores(player) = overall_scores(player) + score;
    end
    
    % Find the winner of this round
    [max_score, winner] = max(scores);
    
    % Display winner of each round
    fprintf('\nWinner of round %d: Player %d\n', round, winner);
    
    % Reset player scores for the next round
    scores = zeros(1, num_players);
end

% Find the overall winner(s)
[~, overall_winners] = max(overall_scores);

% Display overall winner(s)
if length(overall_winners) == 1
    fprintf('\nOverall winner: Player %d\n', overall_winners);
else
    fprintf('\nOverall winners: ');
    for i = 1:length(overall_winners)
        fprintf('Player %d ', overall_winners(i));
    end
    fprintf('\n');
end
