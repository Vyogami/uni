from collections import deque

def water_jug_problem(jug1_capacity, jug2_capacity, target_amount):
    visited_states = set()
    queue = deque([(0, 0, [])])

    while queue:
        current_state = queue.popleft()
        jug1, jug2, steps = current_state

        if current_state[:2] in visited_states:
            continue

        visited_states.add(current_state[:2])

        if jug1 == target_amount or jug2 == target_amount:
            return steps

        queue.append((jug1_capacity, jug2, steps + [(jug1, jug2, "Fill jug 1")]))
        queue.append((jug1, jug2_capacity, steps + [(jug1, jug2, "Fill jug 2")]))
        queue.append((0, jug2, steps + [(jug1, jug2, "Empty jug 1")]))
        queue.append((jug1, 0, steps + [(jug1, jug2, "Empty jug 2")]))
        pour_amount = min(jug1, jug2_capacity - jug2)
        queue.append((jug1 - pour_amount, jug2 + pour_amount, steps + [(jug1, jug2, f"Pour {pour_amount} from jug 1 to jug 2")]))
        pour_amount = min(jug2, jug1_capacity - jug1)
        queue.append((jug1 + pour_amount, jug2 - pour_amount, steps + [(jug1, jug2, f"Pour {pour_amount} from jug 2 to jug 1")]))

    return None  

jug1_capacity = int(input("Enter the capacity of the first jug: "))
jug2_capacity = int(input("Enter the capacity of the second jug: "))
target_amount = int(input("Enter the desired amount of water: "))

result = water_jug_problem(jug1_capacity, jug2_capacity, target_amount)

if result:
    print(f"Minimum number of steps: {len(result)}")
    print("Steps:")
    for step in result:
        print(step)
else:
    print("No solution found.")