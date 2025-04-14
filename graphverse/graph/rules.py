import random
from abc import ABC, abstractmethod


def define_all_rules(
    n,
    num_ascenders,
    num_descenders,
    num_evens,
    num_odds,
    num_repeaters,
    repeater_min_steps,
    repeater_max_steps,
):
    """
    Define all rule vertices while ensuring each vertex is assigned at most one rule.
    """
    available_vertices = set(range(n))

    # Define ascenders and descenders
    ascenders, descenders = define_ascenders_descenders(
        n, num_ascenders, num_descenders, available_vertices
    )

    # Define evens and odds
    evens, odds = define_evens_odds(n, num_evens, num_odds, available_vertices)

    # Define repeaters
    repeaters = define_repeaters(
        n, num_repeaters, repeater_min_steps, repeater_max_steps, available_vertices
    )

    return ascenders, descenders, evens, odds, repeaters


def define_ascenders_descenders(n, num_ascenders, num_descenders, available_vertices):
    """
    Define ascender and descender vertices while ensuring they are not already assigned to another rule.
    Both types are selected from vertices near the middle of the range.
    """
    mid = n // 2
    range_start = int(mid * 0.9)
    range_end = int(mid * 1.1)
    candidates = [v for v in available_vertices if range_start <= v <= range_end]

    ascenders = set(random.sample(candidates, k=min(num_ascenders, len(candidates))))
    available_vertices -= ascenders

    descenders = set(
        random.sample(
            [v for v in candidates if v in available_vertices],  # Recheck availability
            k=min(num_descenders, len(available_vertices)),
        )
    )
    available_vertices -= descenders

    return ascenders, descenders


def define_evens_odds(n, num_evens, num_odds, available_vertices):
    """
    Randomly select even and odd vertices that are not already assigned to another rule.
    """
    even_candidates = [v for v in available_vertices if v % 2 == 0]
    odd_candidates = [v for v in available_vertices if v % 2 != 0]

    evens = set(random.sample(even_candidates, k=min(num_evens, len(even_candidates))))
    available_vertices -= evens

    odds = set(
        random.sample(
            [
                v for v in odd_candidates if v in available_vertices
            ],  # Recheck availability
            k=min(num_odds, len(available_vertices)),
        )
    )
    available_vertices -= odds

    return evens, odds


def define_repeaters(
    n, num_repeaters, repeater_min_steps, repeater_max_steps, existing_rule_vertices
):
    """
    Define repeater vertices and their repetition steps while ensuring they are not already assigned to another rule.
    """
    available_vertices = set(range(n)) - existing_rule_vertices
    repeaters = {}
    for _ in range(num_repeaters):
        if available_vertices:
            v = random.choice(list(available_vertices))
            available_vertices.remove(v)
            repeaters[v] = random.randint(repeater_min_steps, repeater_max_steps)
    return repeaters


def check_rule_compliance(walk, ascenders, descenders, evens, odds, repeaters):
    """Check if a given walk complies with all rules."""
    for i, v in enumerate(walk):
        if v in ascenders:
            if any(walk[j] <= v for j in range(i + 1, len(walk))):
                return False
        if v in descenders:
            if any(walk[j] >= v for j in range(i + 1, len(walk))):
                return False
        if v in evens:
            if any(walk[j] % 2 != 0 for j in range(i + 1, len(walk))):
                return False
        if v in odds:
            if any(walk[j] % 2 == 0 for j in range(i + 1, len(walk))):
                return False
        if v in repeaters:
            indices = [i for i, x in enumerate(walk) if x == v]
            for j in range(len(indices) - 1):
                if indices[j + 1] - indices[j] != repeaters[v]:
                    return False
    return True


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


class DescenderRule(Rule):
    def __init__(self, descenders):
        self.member_nodes = descenders

    def apply(self, graph, walk):
        walk = [int(item) for item in walk]
        for i, v in enumerate(walk):
            if v in self.member_nodes:
                if any(walk[j] > v for j in range(i + 1, len(walk))):
                    return False
        return True

    def get_violation_position(self, graph, walk):
        for i in range(len(walk) - 1):
            if walk[i] in self.member_nodes and walk[i + 1] >= walk[i]:
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
                # get all repeaters nodes in walk
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


class OddRule(Rule):
    def __init__(self, odds):
        self.member_nodes = odds

    def apply(self, graph, walk):
        walk = [int(item) for item in walk]
        for i, v in enumerate(walk):
            if v in self.member_nodes:
                if any(walk[j] % 2 == 0 for j in range(i + 1, len(walk))):
                    return False
        return True

    def get_violation_position(self, graph, walk):
        for i in range(len(walk) - 1):
            if walk[i] in self.member_nodes and walk[i + 1] % 2 == 0:
                return i + 1
        return None


class EdgeExistenceRule(Rule):
    def apply(self, walk, graph):
        for i in range(len(walk) - 1):
            if not graph.has_edge(walk[i], walk[i + 1]):
                return False
        return True
