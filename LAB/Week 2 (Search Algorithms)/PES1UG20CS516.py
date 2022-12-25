"""
You can create any other helper funtions.
Do not modify the given functions
"""
import heapq

def A_star_Traversal(cost, heuristic, start_point, goals):
    """
    Perform A* Traversal and find the optimal path 
    Args:
        cost: cost matrix (list of floats/int)
        heuristic: heuristics for A* (list of floats/int)
        start_point: Staring node (int)
        goals: Goal states (list of ints)
    Returns:
        path: path to goal state obtained from A*(list of ints)
    """
    path=[]
    start = [start_point]
    queue_path = [[0+heuristic[start_point], start]]
    while len(queue_path) > 0:
        curr_cost, curr_path = queue_path.pop(0)
        n = curr_path[-1]
        curr_cost -= heuristic[n]
        if n in goals:
            return curr_path
        path.append(n)
        children = [i for i in range(len(cost[0]))
                    if cost[n][i] not in [0, -1]]
        for i in children:
            new_curr_path = curr_path + [i]
            new_path_cost = curr_cost + cost[n][i] + heuristic[i]
            if i not in path and new_curr_path not in [i[1] for i in queue_path]:
                queue_path.append((new_path_cost, new_curr_path))
                queue_path = sorted(queue_path, key=lambda x: (x[0], x[1]))
            elif new_curr_path in [i[1] for i in queue_path]:
                index = search_q(queue_path, new_curr_path)
                queue_path[index][0] = min(queue_path[index][0], new_path_cost)
                queue_path = sorted(queue_path, key=lambda x: (x[0], x[1]))

    return path

def DFS_Traversal(cost, start_point, goals):
    """
    Perform DFS Traversal and find the optimal path 
        cost: cost matrix (list of floats/int)
        start_point: Staring node (int)
        goals: Goal states (list of ints)
    Returns:
        path: path to goal state obtained from DFS(list of ints)
    """
    path=[]
    queue_path = []
    n = len(cost)
    queue_path.append(start_point)
    while len(queue_path) != 0:
        curr = queue_path.pop()  
        path.append(curr)
        if curr in goals:
                return path
        for i in range(n - 1, 0, -1):
                if cost[curr][i] != -1 and cost[curr][i] != 0 and (i not in path):
                        queue_path.append(i)
    return path
