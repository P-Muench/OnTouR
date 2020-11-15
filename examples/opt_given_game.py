from src.GameLogic.engine import GameFactory
from src.OnTourSolver.MIP import Optimizer


def main():
    cards, regions, board = GameFactory.read_from_txt("../resources/cards_orig.txt", "../resources/board_orig.txt")

    drawn_cards, rolled_dices = read_rounds_from_file("given_game.txt", cards)

    opt = Optimizer(board, drawn_cards, rolled_dices)
    opt.run()


def read_rounds_from_file(filename, cards):
    drawn_cards = []
    rolled_dices = []

    card_by_state = {card.state: card for card in cards}

    txt = open(filename, "r").read()
    for line in txt.splitlines():
        tmp_states, tmp_dices = line.split("(")
        states = [card_by_state[s.strip()] for s in tmp_states.split() if s]
        dices = [int(d) for d in tmp_dices.rstrip("\n )").split(",") if d]
        drawn_cards.append(states)
        rolled_dices.append(dices)
    return drawn_cards, rolled_dices


if __name__ == '__main__':
    main()
