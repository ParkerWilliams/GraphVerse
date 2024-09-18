import random
from .rules import Rule

def check_rule_compliance(graph, walk, rules):
    return all(rule.apply(graph, walk) for rule in rules)

def generate_valid_walk(graph, start_vertex, min_length, max_length, rules, max_attempts=10):
    target_length = random.randint(min_length, max_length)
    walk = [start_vertex]
    attempts = 0
    
    while len(walk) < target_length:
        valid_neighbors = [
            neighbor for neighbor in graph.nodes()
            if check_rule_compliance(walk + [neighbor], rules)
        ]
        
        if not valid_neighbors:
            attempts += 1
            
            if attempts >= max_attempts:
                walk = [start_vertex]
                attempts = 0
            else:
                walk.pop()
        else:
            next_vertex = random.choice(valid_neighbors)
            
            if not graph.has_edge(walk[-1], next_vertex):
                graph.add_edge(walk[-1], next_vertex)
            
            walk.append(next_vertex)
    
    if len(walk) >= min_length:
        return walk
    else:
        return None

def generate_multiple_walks(graph, num_walks, min_length, max_length, rules):
    walks = []
    attempts = 0
    max_attempts = 10
    
    while len(walks) < num_walks:
        start_vertex = random.choice(list(graph.nodes()))
        walk = generate_valid_walk(graph, start_vertex, min_length, max_length, rules)
        
        if walk:
            walks.append(walk)
            attempts = 0
        else:
            attempts += 1
            
            if attempts >= max_attempts:
                attempts = 0
    
    return walks
