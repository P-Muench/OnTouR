from typing import Set, List, Dict, Iterable
import networkx as nx
import random
from abc import ABC, abstractmethod
import re


class Board:

    def __init__(self):
        self.game_graph = nx.Graph()

    def add_node(self, node):
        self.game_graph.add_node(node)

    def add_edge(self, edge_u, edge_v):
        self.game_graph.add_edge(edge_u, edge_v)

    @property
    def states(self):
        return set(self.game_graph.nodes)

    def adjacent_states(self, state):
        if state not in self.states:
            return KeyError(f"State {state} not on board")
        return set(self.game_graph.adj[state])


class Card:

    def __init__(self, state_abbr, state_name, region):
        self._state_abbr = state_abbr
        self._state_name = state_name
        self.region = region

    def __hash__(self):
        return hash((self._state_abbr, self._state_name))

    def __repr__(self):
        return f"Card({self._state_abbr}, {self._state_name}, {self.region})"

    def __str__(self):
        return f"({self._state_abbr}, {self._state_name}, {self.region.name})"

    @property
    def state(self):
        return self._state_abbr


class Region:

    def __init__(self, name, states: Iterable[str] = ()):
        self.name = name
        self._states: Set[str] = set(states)

    def add_state(self, state: str):
        self._states.add(state)

    def __hash__(self):
        return hash(("Region", self.name))

    def __repr__(self):
        return f"Region({self.name}, {self.states})"

    def __iter__(self):
        for state in self.states:
            yield state

    @property
    def states(self) -> Set[str]:
        return set(self._states)


class Deck(ABC):
    def __init__(self, cards: Iterable[Card]):
        self._cards: List[Card] = list(cards)

    @abstractmethod
    def draw_cards(self, num_cards: int):
        pass


class RandomDeck(Deck):

    def __init__(self, cards: Iterable[Card]):
        super().__init__(cards)

    def shuffle(self):
        indices = list(range(len(self._cards)))
        random.shuffle(indices)

        self._shuffled_cards = [self._cards[i] for i in indices]

    def draw_cards(self, num_cards, remove_from_deck=False):
        if num_cards > len(self._cards) or num_cards < 0:
            raise ValueError(f"Cannot draw below 0 or above {len(self._cards)} cards.")
        try:
            shuffled_cards = self._shuffled_cards
        except AttributeError:
            raise RuntimeError("Deck must be shuffled first")
        to_return = shuffled_cards[:num_cards]
        self._shuffled_cards = shuffled_cards[num_cards:]
        if len(to_return) == len(shuffled_cards):
            # if all cards are drawn -> reshuffle
            self.shuffle()
        if len(to_return) < num_cards:
            # if not enough cards are drawn, draw from newly shuffled, exclude one currently in hand
            for card in to_return:
                self._shuffled_cards.remove(card)
            to_return += self.draw_cards(num_cards - len(to_return))
        if remove_from_deck:
            for c in to_return:
                try:
                    self._shuffled_cards.remove(c)
                    self._cards.remove(c)
                except:
                    pass
        return to_return


class GameFactory:

    @staticmethod
    def read_from_txt(fp_cards, fp_board):
        tmp_cards = GameFactory._read_cards_from_txt(fp_cards)
        cards, regions, board = GameFactory._read_board_from_txt(fp_board, tmp_cards)
        # sanity check
        for card in cards:
            if type(card.region) is not Region:
                raise AssertionError(f"Sanity check failed. No region provided for card {card.name}")
        card_wo_node = set(card._state_abbr for card in cards) - board.states
        node_wo_card = board.states - set(card._state_abbr for card in cards)
        if len(card_wo_node):
            raise AssertionError(f"Cards have no nodes on the board: {card_wo_node}")
        if len(node_wo_card):
            raise AssertionError(f"Nodes have no cards in the deck: {node_wo_card}")
        return cards, regions, board

    @staticmethod
    def _read_cards_from_txt(fp_cards):
        txt = open(fp_cards).read()
        tmp_cards = dict()
        for line in txt.split("\n"):
            abbr, name, region = line.split()
            tmp_cards[abbr] = Card(abbr, name, region)
        return tmp_cards

    @staticmethod
    def _read_board_from_txt(fp_board, tmp_cards):
        txt = open(fp_board).read()
        matches = re.finditer(r"--\w*--[^-]*", txt, re.MULTILINE)

        states_txt = ""
        regions_txt = ""
        connections_txt = ""

        for match in matches:
            group = match.group()
            if group.startswith("--REGIONS--\n"):
                regions_txt = group[len("--REGIONS--\n"):]
            elif group.startswith("--CONNECTIONS--\n"):
                connections_txt = group[len("--CONNECTIONS--\n"):]
            elif group.startswith("--STATES--\n"):
                states_txt = group[len("--STATES--\n"):]
        if not all((states_txt, regions_txt, connections_txt)):
            raise ValueError("Text does not specify regions, connections and states.")

        states = set(states_txt.splitlines())
        connections = set(tuple(line.split()) for line in connections_txt.splitlines())
        regions = set()
        for line in regions_txt.splitlines():
            name, region_states = line.split(":")
            reg = Region(name)
            for s in region_states.split():
                if s:
                    reg.add_state(s)
                    tmp_cards[s].region = reg

        board = Board()
        for state in states:
            board.add_node(state)
        for (u, v) in connections:
            board.add_edge(u, v)
        return tmp_cards.values(), regions, board


def roll_dices(num_sides=10):
    return random.randint(0, num_sides - 1), random.randint(0, num_sides - 1)
