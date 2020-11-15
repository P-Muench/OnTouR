from gurobipy import *
from typing import List, Dict, Iterable, Set, Optional, Any
from itertools import product, combinations
from collections import defaultdict
from src.GameLogic.engine import Card

STAR_SYMBOL = "☆"
X_SYMBOL = "x"


class Optimizer:
    DICE_SIDES = list(range(10))

    def __init__(self, board, drawn_cards, rolled_dices):
        if len(rolled_dices) != len(drawn_cards):
            raise ValueError("Number of rounds differ for rolled dices and drawn cards")
        self._rolled_dices = list(rolled_dices)
        self._board = board
        self._states = set(board.states)
        self._drawn_cards = list(drawn_cards)
        self._symbols: Set[Any] = set()
        self._moves_per_round: Dict[int, Set[Move]] = dict()
        self._rounds: List[int] = list(range(len(self._rolled_dices)))
        self._path_positions = list(range(len(self.states)))
        self._possible_states_per_round: Dict[int, Set[Any]] = dict()
        self._num_moves_per_round: Dict[int, int] = dict()

    def _optimize(self, use_indicator_variables):
        m = Model()

        ub_symbols = (11*(max(Optimizer.DICE_SIDES)))
        # Variables
        choose_move = m.addVars(((r, move) for r in self._rounds for move in self._moves_per_round[r]),
                               vtype=GRB.BINARY, name="choose_move")
        symbol_to_state = m.addVars(self.symbols, self.states, vtype=GRB.BINARY, name="symbol_to_state")
        circled_state = m.addVars(self.states, vtype=GRB.BINARY, name="circled_state")
        state_to_path = m.addVars(self.states, self._path_positions, vtype=GRB.BINARY, name="state_to_path")
        if use_indicator_variables:
            symbol_to_path = m.addVars(self.symbols, self._path_positions, vtype=GRB.BINARY, name="symbol_to_path")
        else:
            number_to_path = m.addVars(self._path_positions, vtype=GRB.CONTINUOUS, ub=ub_symbols, name="number_to_path")
        circled_in_path = m.addVars(self.states, vtype=GRB.BINARY, name="circled_in_path")
        is_chosen_until_round = m.addVars(self._rounds, self.states, vtype=GRB.BINARY, name="is_chosen_until_round")
        all_reqs = set()
        for r in self._rounds:
            for move in self._moves_per_round[r]:
                if move.has_x_symbol():
                    all_reqs.add((r, move.required_states))
        is_req_fulfilled = m.addVars(all_reqs, vtype=GRB.BINARY, name="is_req_fulfilled")

        # In every round exactly one move must be chosen
        m.addConstrs(quicksum(choose_move[r, move] for move in self._moves_per_round[r]) == 1 for r in self._rounds)

        # determine the symbol chosen for a state by the chosen move
        m.addConstrs(symbol_to_state[sy, st]
                     ==
                     quicksum(choose_move[r, move] for r in self._rounds for move in
                              self._moves_by_st_sy[r, st, sy])
                     for sy in self.symbols for st in self.states)

        # make sure every state gets exactly one symbol assigned
        m.addConstrs(quicksum(symbol_to_state[sy, st] for sy in self.symbols) == 1 for st in self.states)

        # determine if a state is encircled
        m.addConstrs(circled_state[st]
                     ==
                     quicksum(choose_move[r, move] * move.is_state_circled(st) for r in self._rounds for move in
                              self._moves_by_st[r, st])
                     for st in self.states)

        # Helper: Each state must be chosen in exactly one round
        m.addConstrs(quicksum(choose_move[r, move] for r in self._rounds for move in self._moves_by_st[r, st])
                     == 1
                     for st in self.states)

        is_chosen_until_round_rhs = {(0, st): 0 for st in self.states}
        for r in self._rounds[1:]:
            for st in self.states:
                is_chosen_until_round_rhs[r, st] = is_chosen_until_round_rhs[r - 1, st] \
                                         + quicksum(choose_move[r, move] for move in self._moves_by_st[r, st])
        m.addConstrs(is_chosen_until_round[r, st] == is_chosen_until_round_rhs[r, st] for r in self._rounds for st in self.states)

        # an assignment with an "x" can only be chosen, if every required state is already filled
        # m.addConstrs(choose_move[r, move]
        #              <=
        #              quicksum(is_chosen_until_round[r - 1, st] for st in move.required_states) / len(move.required_states)
        #              for r in self._rounds for move in self._moves_per_round[r] if r >= 2 and move.has_x_symbol())
        for r, reqs in all_reqs:
            m.addGenConstrAnd(is_req_fulfilled[r, reqs], [is_chosen_until_round[r - 1, st] for st in reqs], "req_fulfilled[%s,%s]" % (r, reqs))
        m.addConstrs(choose_move[r, move]
                     <=
                     is_req_fulfilled[r, move.required_states]
                     for r in self._rounds for move in self._moves_per_round[r] if r >= 2 and move.has_x_symbol())

        # every position in path can be filled by at most one state
        m.addConstrs(quicksum(state_to_path[st, pos] for st in self.states) <= 1 for pos in self._path_positions)

        # every state can be added at most once to path
        # m.addConstrs(quicksum(state_to_path[st, pos] for pos in self._path_positions) <= 1 for st in self.states)
        for st in self.states:
            m.addSOS(GRB.SOS_TYPE1, [state_to_path[st, pos] for pos in self._path_positions], list(self._path_positions))

        if use_indicator_variables:
            # at each path position there can be at most one symbol
            m.addConstrs(quicksum(symbol_to_path[sy, pos] for sy in self.symbols) <= 1 for pos in self._path_positions)

            # determine which symbol is at which position in path (if symbol is "star" then solver can decide symbol)
            m.addConstrs(symbol_to_path[sy, pos] >= state_to_path[st, pos] + symbol_to_state[sy, st] - 1
                         for sy in self.symbols for st in self.states for pos in self._path_positions
                         if not sy == STAR_SYMBOL)

            # assign any symbol on "star"s on path
            m.addConstrs(quicksum(symbol_to_path[sy, pos] for sy in self.symbols - {STAR_SYMBOL})
                         >=
                         state_to_path[st, pos] + symbol_to_state[STAR_SYMBOL, st] - 1
                         for st in self.states for pos in self._path_positions)

            # only assign symbol to path if a state is assigned
            m.addConstrs(quicksum(symbol_to_path[sy, pos] for sy in self.symbols)
                         <=
                         quicksum(state_to_path[st, pos] for st in self.states)
                         for pos in self._path_positions)
        else:
            # determine symbol at path positions
            m.addConstrs(number_to_path[pos] >= - ub_symbols * (1 - state_to_path[st, pos]) + sy * symbol_to_state[sy, st]
                         for sy in self.symbols for st in self.states for pos in self._path_positions
                         if not sy in [X_SYMBOL, STAR_SYMBOL])
            m.addConstrs(number_to_path[pos] <= 2 * ub_symbols * (1 - state_to_path[st, pos]) + sy * symbol_to_state[sy, st] + ub_symbols * (1 - symbol_to_state[sy, st])
                         for sy in self.symbols for st in self.states for pos in self._path_positions
                         if not sy in [X_SYMBOL, STAR_SYMBOL])

            # forbid states with "x" in path
            m.addConstrs(
                number_to_path[pos] >= - (ub_symbols + 1) * (1 - state_to_path[st, pos]) + (ub_symbols + 1) * symbol_to_state[X_SYMBOL, st]
                for st in self.states for pos in self._path_positions)

        # adjacent positions in path must be adjacent in game graph
        m.addConstrs(state_to_path[st, pos]
                     <=
                     quicksum(state_to_path[st_, pos - 1] for st_ in self._board.adjacent_states(st))
                     for st in self.states for pos in self._path_positions if pos >= 1)

        if use_indicator_variables:
            # adjacent positions in path must be of ascending symbols
            m.addConstrs(symbol_to_path[sy1, pos] <= 1 - symbol_to_path[sy2, pos - 1] * (sy2 >= sy1)
                         for pos in self._path_positions for sy1, sy2 in product(self.symbols, self.symbols)
                         if not any(sy in [X_SYMBOL, STAR_SYMBOL] for sy in (sy1, sy2)) and pos >= 1)

            # "x" cannot be added to path
            m.addConstrs(symbol_to_path[X_SYMBOL, pos] == 0 for pos in self._path_positions)
        else:
            # adjacent positions in path must be of ascending symbols
            m.addConstrs(number_to_path[pos] >= number_to_path[pos - 1]
                         for pos in self._path_positions if pos >= 1)

        # path must start at 0 and cannot skip positions
        m.addConstrs(quicksum(state_to_path[st, pos] for st in self.states)
                     >=
                     quicksum(state_to_path[st, pos + 1] for st in self.states)
                     for pos in self._path_positions[:-1])

        # determine if state in path is also circled by preventing circling if state not in path or state no circled
        m.addConstrs(circled_in_path[st] <= quicksum(state_to_path[st, pos] for pos in self._path_positions) for st in self.states)
        m.addConstrs(circled_in_path[st] <= circled_state[st] for st in self.states)

        m.setObjective(quicksum(circled_in_path[st] for st in self.states)
                       + quicksum(state_to_path[st, pos] for pos in self._path_positions for st in self.states)
                       , GRB.MAXIMIZE)

        m.setParam("TimeLimit", 5400)
        m.setParam("Presolve", 2)
        # m.setParam("Heuristics", 0.2)
        # m.setParam("MIPFocus", 1)

        m.update()
        for v in m.getVars():
            v.VarName = v.VarName.replace(" ", "")
        m.update()

        try:
            m.setParam("OutputFlag", 0)
            m.read("test.sol")
        except:
            pass
        finally:
            m.setParam("OutputFlag", 1)

        # ID and CO, 85 and 58
        # ass_1 = Assignment(85, "ID", True)
        # ass_2 = Assignment(58, "CO", True)
        #
        # m.addConstr(choose_move[2, AssignmentCollection([ass_1, ass_2])] == 1)

        m.optimize()

        if m.status == GRB.INFEASIBLE:
            m.computeIIS()
            m.write("test.ilp")
        else:
            m.write("test.sol")
            state_symbols = dict()
            for (sy, st) in symbol_to_state:
                if symbol_to_state[sy, st].x > .5:
                    state_symbols[st] = sy

            path_states = []
            path_real_symbols = []
            path_repl_symbols = []
            for pos in self._path_positions:
                for st in self.states:
                    if state_to_path[st, pos].x > .5:
                        for sy in self.symbols:
                            if symbol_to_state[sy, st].x > .5:
                                path_real_symbols.append(str(sy))
                        if circled_state[st].x > .5:
                            st = "(" + st + ")"
                        path_states.append(st)
                        if not use_indicator_variables:
                            path_repl_symbols.append(str(int(number_to_path[pos].x + .2)))
                if use_indicator_variables:
                    for sy in self.symbols:
                        if symbol_to_path[sy, pos].x > .5:
                            path_repl_symbols.append(str(sy))
            print("Best found path:")
            print("States: " + " -> ".join(path_states))
            print("Symbols: " + " -> ".join(path_real_symbols))
            print("Numbers: " + " -> ".join(path_repl_symbols))

            for (st, sy) in state_symbols.items():
                print(f"{sy}\t->\t{st}")
            # for key in choose_move:
            #     if choose_move[key].x > .5:
            #         print(key)

    def _preprocess(self):
        self._symbols = {10 * digit_1 + digit_2
                         for (digit_1, digit_2) in product(Optimizer.DICE_SIDES, Optimizer.DICE_SIDES)
                         if digit_1 != digit_2}.union(STAR_SYMBOL).union(X_SYMBOL)

        for r in self._rounds:
            if len(set(self._rolled_dices[r])) == 1:
                symbols_from_dices = [STAR_SYMBOL]
            else:
                digit_1, digit_2 = self._rolled_dices[r]
                symbols_from_dices = [10 * digit_1 + digit_2, 10 * digit_2 + digit_1]

            # if dices are the same, then only a "star" can be placed -> 1 state more filled
            self._num_moves_per_round[r] = len(symbols_from_dices)
            if r in [0, 1]:
                # initializing in first 2 rounds
                card1, card2 = self._drawn_cards[r]
                lower_symbol, upper_symbol = min(symbols_from_dices), max(symbols_from_dices)
                # lower symbol is placed on first card, upper symbol on second
                ass_1 = Assignment(lower_symbol, card1.state, circled=True)
                ass_2 = Assignment(upper_symbol, card2.state, circled=True)
                self._moves_per_round[r] = {Move([ass_1, ass_2])}
            elif sum(self._num_moves_per_round[r_] for r_ in range(r)) < len(self.states) - 2:
                # every round if more than 2 states are left
                self._moves_per_round[r] = create_moves_from_round(self._drawn_cards[r], symbols_from_dices, self.states, include_x_moves=True)
            else:
                # there are only 2 states left to be marked
                if len(symbols_from_dices) == 1:
                    # if a double is rolled, you can possibly assign a star to every state
                    self._moves_per_round[r] = {Move([Assignment(STAR_SYMBOL, st, circled=False)]) for st in self.states}
                else:
                    if sum(self._num_moves_per_round[r_] for r_ in range(r)) == len(self.states) - 2:
                        # if there are two states left
                        self._moves_per_round[r] = set()
                        # if two different numbers are rolled, each one can be assigned to every state
                        for st1, st2 in product(self.states, self.states):
                            if st1 != st2:
                                move1 = Move([Assignment(symbols_from_dices[0], st1, circled=False),
                                              Assignment(symbols_from_dices[1], st2, circled=False)])
                                move2 = Move([Assignment(symbols_from_dices[1], st1, circled=False),
                                              Assignment(symbols_from_dices[0], st2, circled=False)])
                                self._moves_per_round[r].add(move1)
                                self._moves_per_round[r].add(move2)
                    else:
                        # there is only one state left, one of the symbols can be placed on any state
                        self._moves_per_round[r] = {
                            Move([Assignment(sy, st, circled=False)])
                            for st in self.states for sy in self.symbols if not sy in [X_SYMBOL, STAR_SYMBOL]}

            self._possible_states_per_round[r] = set(
                state for move in self._moves_per_round[r] for state in move.get_states())

        total_moves = sum(self._num_moves_per_round.values())
        if total_moves < len(self.states) or total_moves > len(self.states) + 1:
            raise ValueError("Not enough rounds to cover every state")

        # preprocessing makes constraint generation faster
        self._moves_by_st_sy = defaultdict(list)
        self._moves_by_st = defaultdict(list)
        for r in self._rounds:
            for move in self._moves_per_round[r]:
                for ass in move:
                    self._moves_by_st_sy[r, ass.state, ass.symbol].append(move)
                    self._moves_by_st[r, ass.state].append(move)

    def run(self, use_indicator_variables=False):
        self._preprocess()
        self._optimize(use_indicator_variables)

    @property
    def rolled_dices(self):
        return list(self._rolled_dices)

    @property
    def drawn_cards(self):
        return list(self._drawn_cards)

    @property
    def symbols(self):
        return set(self._symbols)

    @property
    def states(self):
        return set(self._states)


class Assignment:

    def __init__(self, symbol, state, circled: bool = False, from_card=None):
        self._state = state
        self._symbol = symbol
        self._circled = circled
        self.from_card = from_card

    @property
    def state(self):
        return self._state

    @property
    def symbol(self):
        return self._symbol

    @property
    def circled(self):
        return self._circled

    def __lt__(self, other):
        if type(other) is Assignment:
            return (str(self._symbol), self._state, self._circled) < (str(other._symbol), other._state, other._circled)
        else:
            raise NotImplementedError

    def __hash__(self):
        return hash((self._state, self._symbol, self._circled))

    def __repr__(self):
        return f"Assignment({self._symbol}, {self._state}, {self._circled})"

    def __eq__(self, other):
        if type(other) is Assignment:
            return self.state == other.state and self.symbol == other.symbol and self.circled == other.circled
        else:
            raise NotImplementedError


class Move:

    def __init__(self, iterable: Iterable[Assignment] = (), required_states=()):
        self._assignments: Set[Assignment] = set(iterable)
        self.required_states = frozenset(required_states)

    def add_assignment(self, assignment: Assignment):
        self._assignments.add(assignment)

    def is_state_circled(self, state) -> bool:
        return any(ass.circled and ass.state == state for ass in self._assignments)

    def __contains__(self, item):
        if type(item) is Assignment:
            return item in self._assignments
        elif type(item) is tuple and len(item) == 2:
            return any(ass.symbol == item[0] and ass.state == item[1] for ass in self._assignments)
        else:
            raise ValueError("Item must be either of Type Assignment or tuple (symbol, state).")

    def has_x_symbol(self) -> bool:
        return sum(ass.symbol == X_SYMBOL for ass in self._assignments)

    def get_states(self, include_xs=False) -> Set[Any]:
        return set(ass.state for ass in self._assignments if (not ass.symbol == X_SYMBOL) or include_xs)

    def __len__(self) -> int:
        return len(self._assignments)

    def __hash__(self):
        return hash(tuple(sorted(self._assignments)))

    def __iter__(self):
        for ass in self._assignments:
            yield ass

    def __repr__(self):
        return "AssignmentCollection((" + ",".join(repr(ass) for ass in sorted(self._assignments)) + "))"

    def __eq__(self, other):
        if type(other) is Move:
            return self._assignments == other._assignments
        else:
            raise NotImplementedError


def create_moves_from_round(cards: Set[Card], symbols, all_states, include_x_moves) -> Set[Move]:
    all_possible_moves = set()
    symbols = set(symbols)
    cards = set(cards)
    all_states = set(all_states)
    for sy in symbols:
        for card in cards:
            for related_states in card.region:
                if len(symbols) == 1:
                    move = Move()
                    ass = Assignment(sy, related_states, circled=(card.state == related_states), from_card=card)
                    move.add_assignment(ass)
                    all_possible_moves.add(move)
                else:
                    for sy2 in (symbols - {sy}):
                        for card2 in cards - {card}:
                            for related_states2 in card2.region:
                                move = Move()
                                ass = Assignment(sy, related_states, circled=(card.state == related_states), from_card=card)
                                ass2 = Assignment(sy2, related_states2, circled=(card2.state == related_states2), from_card=card2)
                                move.add_assignment(ass)
                                move.add_assignment(ass2)
                                all_possible_moves.add(move)
    if include_x_moves:
        # create moves with x_symbols
        states_covered_by_cards = {related_state for card in cards for related_state in card.region}
        states_not_covered_by_cards = all_states - states_covered_by_cards
        if len(symbols) == 1:
            # if there is only one symbol to set: X can be at most in one of the states not covered by all the cards regions
            for st in states_not_covered_by_cards:
                ass = Assignment(X_SYMBOL, st, circled=False, from_card=None)
                all_possible_moves.add(Move([ass], required_states=states_covered_by_cards))
        else:
            # if there are two symbols to set:
            for card in cards:
                states_covered_by_other_cards = {related_state2 for card2 in cards - {card} for related_state2 in
                                                 card2.region}
                for sy in symbols:
                    for related_state in card.region:
                        # 1. take any card, set one of the symbols
                        ass1 = Assignment(sy, card.state, circled=(card.state == related_state), from_card=card)
                        for st in all_states - states_covered_by_other_cards:
                            # 2. X can be at most in one of the states not covered the remaining cards regions
                            ass2 = Assignment(X_SYMBOL, st, circled=False, from_card=None)
                            all_possible_moves.add(Move([ass1, ass2], required_states=states_covered_by_other_cards))
            # 3. It is possible that two x's must be set. This is only possible in states not covered by any of the cards regions
            for st1, st2 in product(states_not_covered_by_cards, states_not_covered_by_cards):
                if st1 != st2:
                    all_possible_moves.add(Move([
                        Assignment(X_SYMBOL, st1, False, from_card=None),
                        Assignment(X_SYMBOL, st2, False, from_card=None),
                    ], required_states=states_covered_by_cards))
    return all_possible_moves
