import random
from abc import ABC, abstractmethod


def define_all_rules(n, num_ascenders, num_evens, num_repeaters, repeater_min_steps, repeater_max_steps):
    """
    Define all rule vertices while ensuring each vertex is assigned at most one rule.
    """
    available_vertices = set(range(n))

    # Define ascenders
    ascenders = define_ascenders(n, num_ascenders, available_vertices)
    available_vertices -= ascenders
    
    # Define evens
    evens = define_evens(n, num_evens, available_vertices)
    available_vertices -= evens
    
    # Define repeaters
    repeaters = define_repeaters(n, num_repeaters, repeater_min_steps, repeater_max_steps, available_vertices)

    return ascenders, evens, repeaters


def define_ascenders(n, num_ascenders, available_vertices):
    """
    Define ascender vertices while ensuring they are not already assigned to another rule.
    Selected from vertices near the middle of the range.
    """
    mid = n // 2
    range_start = int(mid * 0.9)
    range_end = int(mid * 1.1)
    candidates = [v for v in available_vertices if range_start <= v <= range_end]
    
    return set(random.sample(candidates, k=min(num_ascenders, len(candidates))))


def define_evens(n, num_evens, available_vertices):
    """
    Randomly select even vertices that are not already assigned to another rule.
    """
    even_candidates = [v for v in available_vertices if v % 2 == 0]
    return set(random.sample(even_candidates, k=min(num_evens, len(even_candidates))))


def define_repeaters(
    n, num_repeaters, repeater_min_steps, repeater_max_steps, available_vertices
):
    """
    Define repeater vertices and their repetition steps while ensuring they are not already assigned to another rule.
    """
    repeaters = {}
    for _ in range(num_repeaters):
        if available_vertices:
            v = random.choice(list(available_vertices))
            available_vertices.remove(v)
            repeaters[v] = random.randint(repeater_min_steps, repeater_max_steps)
    return repeaters


class Rule(ABC):
    @abstractmethod
    def apply(self, walk, graph):
        """
        Check if the rule is satisfied for the given walk.

        :param walk: List of vertices representing the walk
        :param graph: The graph on which the walk is performed
        :return: True if the rule is satisfied, False otherwise
        """
        pass


class AscenderRule(Rule):
    def __init__(self, ascenders):
        self.member_nodes = ascenders

    def apply(self, graph, walk):
        walk = [int(item) for item in walk]
        for i, v in enumerate(walk):
            if v in self.member_nodes:
                if any(walk[j] < v for j in range(i + 1, len(walk))):
                    return False
        return True

    def get_violation_position(self, graph, walk):
        for i in range(len(walk) - 1):
            if walk[i] in self.member_nodes and walk[i + 1] <= walk[i]:
                return i + 1
        return None


class EvenRule(Rule):
    def __init__(self, evens):
        self.member_nodes = evens

    def apply(self, graph, walk):
        walk = [int(item) for item in walk]
        for i, v in enumerate(walk):
            if v in self.member_nodes:
                if any(walk[j] % 2 != 0 for j in range(i + 1, len(walk))):
                    return False
        return True

    def get_violation_position(self, graph, walk):
        for i in range(len(walk) - 1):
            if walk[i] in self.member_nodes and walk[i + 1] % 2 != 0:
                return i + 1
        return None

class RepeaterRule(Rule):
    def __init__(self, repeaters):
        self.member_nodes = set(repeaters.keys())
        self.members_nodes_dict = repeaters

    def apply(self, graph, walk):
        walk = [int(item) for item in walk]
        for v, k in self.members_nodes_dict.items():
            if v in walk:
                indices = [i for i, x in enumerate(walk) if x == v]
                for i in range(len(indices) - 1):
                    if indices[i + 1] - indices[i] != k:
                        return False
        return True

    def get_violation_position(self, walk):
        for v, k in self.members_nodes_dict.items():
            if v in walk:
                indices = [i for i, x in enumerate(walk) if x == v]
                for i in range(len(indices) - 1):
                    if indices[i + 1] - indices[i] != k:
                        return indices[i + 1]
        return None


def check_rule_compliance(walk, ascenders, evens, repeaters):
    """Check if a given walk complies with all rules."""
    for i, v in enumerate(walk):
        if v in ascenders:
            if any(walk[j] <= v for j in range(i + 1, len(walk))):
                return False
        if v in evens:
            if any(walk[j] % 2 != 0 for j in range(i + 1, len(walk))):
                return False
        if v in repeaters:
            indices = [i for i, x in enumerate(walk) if x == v]
            for j in range(len(indices) - 1):
                if indices[j + 1] - indices[j] != repeaters[v]:
                    return False
    return True


