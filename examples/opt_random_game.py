from src.GameLogic.engine import GameFactory, RandomDeck, roll_dices
from src.OnTourSolver.MIP import Optimizer
import random


if __name__ == '__main__':
    random.seed(1)

    # cards, regions, board = GameFactory.read_from_txt("../resources/cards_very_small.txt", "../resources/board_very_small.txt")
    # cards, regions, board = GameFactory.read_from_txt("../resources/cards_small.txt", "../resources/board_small.txt")
    cards, regions, board = GameFactory.read_from_txt("../resources/cards_orig.txt", "../resources/board_orig.txt")

    deck = RandomDeck(cards)
    deck.shuffle()

    drawn_cards = []
    rolled_dices = []

    covered_states = 0
    draw_round = 0
    while covered_states < len(board.states):
        if draw_round in [0, 1]:
            # initializing rounds
            drawn_cards.append(deck.draw_cards(2, remove_from_deck=True))
            dices = roll_dices()
            while dices[0] == dices[1]:
                # reroll doubles
                dices = roll_dices()
            rolled_dices.append(dices)
            covered_states += 2
        else:
            if covered_states < len(board.states) - 2:
                drawn_cards.append(deck.draw_cards(3))
            else:
                # if only two states are missing, no cards are drawn and symbols can be written to every
                # remaining state
                drawn_cards.append([])
            if drawn_cards[-1] and all(card.region == drawn_cards[-1][0].region for card in drawn_cards[-1]):
                # if all cards have the same region: treat as if double is rolled
                dices = (0, 0)
            else:
                # else roll 2 dices
                dices = roll_dices()
            rolled_dices.append(dices)

            # if same number has been rolled, only one symbol is added to map, otherwise 2
            if dices[0] == dices[1]:
                covered_states += 1
            else:
                covered_states += 2
        print(drawn_cards[-1], rolled_dices[-1])
        draw_round += 1

    opt = Optimizer(board, drawn_cards, rolled_dices)
    opt.run()
