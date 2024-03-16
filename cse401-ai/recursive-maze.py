def dfs_maze_solver(maze, start, end, visited):
    rows, cols = len(maze), len(maze[0])

    def is_valid_move(row, col):
        return 0 <= row < rows and 0 <= col < cols and maze[row][col] == '.' and not visited[row][col]

    def dfs(row, col):
        visited[row][col] = True

        if (row, col) == end:
            return True

        # Explore in all four directions: up, down, left, right
        directions = [(0, -1), (-1, 0), (0, 1), (1, 0)]

        for dr, dc in directions:
            new_row, new_col = row + dr, col + dc
            if is_valid_move(new_row, new_col) and dfs(new_row, new_col):
                return True

        return False

    start_row, start_col = start
    return dfs(start_row, start_col)

# Get user input for maze dimensions
rows = int(input("Enter the number of rows in the maze: "))
cols = int(input("Enter the number of columns in the maze: "))

# Get user input for maze structure
print("Enter the maze structure (use '.' for open path and '#' for wall):")
maze = [input().strip() for _ in range(rows)]

# Get user input for start and end points
start_point = tuple(map(int, input("Enter the start point (row col): ").split()))
end_point = tuple(map(int, input("Enter the end point (row col): ").split()))

# Initialize visited matrix
visited = [[False for _ in range(cols)] for _ in range(rows)]

# Solve the maze
if dfs_maze_solver(maze, start_point, end_point, visited):
    print("\nSolution found:")
    for i in range(rows):
        for j in range(cols):
            if visited[i][j]:
                print('*', end=' ')
            else:
                print(maze[i][j], end=' ')
        print()
else:
    print("No solution found.")
