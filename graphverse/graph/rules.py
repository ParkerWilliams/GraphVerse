import random
from abc import ABC, abstractmethod


def define_all_rules_by_percentage(n, ascender_percent, even_percent, repeater_percent, repeater_min_steps, repeater_max_steps, verbose=False):
    """
    Define all rule vertices using percentages while ensuring each vertex is assigned at most one rule.
    
    Args:
        n: Total number of nodes in the graph
        ascender_percent: Percentage of nodes to be ascenders (0.0 to 100.0)
        even_percent: Percentage of nodes to be even rule nodes (0.0 to 100.0)
        repeater_percent: Percentage of nodes to be repeaters (0.0 to 100.0)
        repeater_min_steps: Minimum steps for repeater rules
        repeater_max_steps: Maximum steps for repeater rules
        verbose: Whether to print detailed information
        
    Returns:
        tuple: (ascenders_set, evens_set, repeaters_dict)
    """
    if verbose:
        print(f"\n  Defining rules by percentage for {n} nodes:")
        print(f"    Ascender percentage: {ascender_percent}%")
        print(f"    Even rule percentage: {even_percent}%")
        print(f"    Repeater percentage: {repeater_percent}%")
    
    # Convert percentages to counts
    num_ascenders = max(0, int(n * ascender_percent / 100))
    num_evens = max(0, int(n * even_percent / 100))
    num_repeaters = max(0, int(n * repeater_percent / 100))
    
    if verbose:
        print(f"    Converted to counts: {num_ascenders} ascenders, {num_evens} evens, {num_repeaters} repeaters")
        total_rule_nodes = num_ascenders + num_evens + num_repeaters
        actual_percentage = (total_rule_nodes / n) * 100
        print(f"    Total rule nodes: {total_rule_nodes}/{n} ({actual_percentage:.1f}%)")
    
    # Use existing function with calculated counts
    ascenders, evens, repeaters = define_all_rules(
        n, num_ascenders, num_evens, num_repeaters, repeater_min_steps, repeater_max_steps
    )
    
    if verbose:
        actual_ascender_percent = (len(ascenders) / n) * 100
        actual_even_percent = (len(evens) / n) * 100
        actual_repeater_percent = (len(repeaters) / n) * 100
        print(f"    Actual percentages achieved:")
        print(f"      Ascenders: {len(ascenders)} nodes ({actual_ascender_percent:.2f}%)")
        print(f"      Even rules: {len(evens)} nodes ({actual_even_percent:.2f}%)")
        print(f"      Repeaters: {len(repeaters)} nodes ({actual_repeater_percent:.2f}%)")
    
    return ascenders, evens, repeaters


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
    def is_satisfied_by(self, walk, graph):
        """
        Check if the rule is satisfied for the given walk.

        :param walk: List of vertices representing the walk
        :param graph: The graph on which the walk is performed
        :return: True if the rule is satisfied, False otherwise
        """
        pass

    @abstractmethod
    def apply(self, walk, graph):
        """
        Alias for is_satisfied_by for backward compatibility.
        """
        return self.is_satisfied_by(walk, graph)


class AscenderRule(Rule):
    def __init__(self, ascenders):
        self.member_nodes = ascenders
        self.is_ascender_rule = True
        self.is_repeater_rule = False
        self.is_even_rule = False

    def is_satisfied_by(self, walk, graph):
        walk = [int(item) for item in walk]
        for i, v in enumerate(walk):
            if v in self.member_nodes:
                if any(walk[j] < v for j in range(i + 1, len(walk))):
                    return False
        return True

    def apply(self, walk, graph):
        return self.is_satisfied_by(walk, graph)

    def get_violation_position(self, graph, walk):
        for i in range(len(walk) - 1):
            if walk[i] in self.member_nodes and walk[i + 1] <= walk[i]:
                return i + 1
        return None


class EvenRule(Rule):
    def __init__(self, evens):
        self.member_nodes = evens
        self.is_even_rule = True
        self.is_ascender_rule = False
        self.is_repeater_rule = False

    def is_satisfied_by(self, walk, graph):
        walk = [int(item) for item in walk]
        for i, v in enumerate(walk):
            if v in self.member_nodes:
                if any(walk[j] % 2 != 0 for j in range(i + 1, len(walk))):
                    return False
        return True

    def apply(self, walk, graph):
        return self.is_satisfied_by(walk, graph)

    def get_violation_position(self, graph, walk):
        for i in range(len(walk) - 1):
            if walk[i] in self.member_nodes and walk[i + 1] % 2 != 0:
                return i + 1
        return None

class RepeaterRule(Rule):
    def __init__(self, repeaters):
        self.member_nodes = set(repeaters.keys())
        self.members_nodes_dict = repeaters
        self.is_repeater_rule = True
        self.is_ascender_rule = False
        self.is_even_rule = False

    def is_satisfied_by(self, walk, graph):
        walk = [int(item) for item in walk]
        for v, k in self.members_nodes_dict.items():
            if v in walk:
                indices = [i for i, x in enumerate(walk) if x == v]
                for i in range(len(indices) - 1):
                    if indices[i + 1] - indices[i] != k:
                        return False
        return True

    def apply(self, walk, graph):
        return self.is_satisfied_by(walk, graph)

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


