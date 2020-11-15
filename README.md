# OnTouR
Library to find optimal moves for a game of OnTour.

Based on the game [OnTour by BoardGameTables](https://www.boardgametables.com/products/on-tour-usa-and-europe "BoardGameTables Homepage") this library implements basic game logic such as cards, regions and the board. While being non-playable it serves as a basis for an algorithm that solves the following problem:

> Given the full information of a game of OnTour (drawn cards and rolled dices) how much points one could get if played perfectly?

It is the 'offline' (full information) counterpart to the regular game of OnTour. OnTour falls into the category of online optimizations. That is optimization problems where not all the information is given in advance.
The algorithm formulates this problem as an integrated Mixed Integer Program (MIP). The game logic is modeled with together with the optimization criteria of find a most-valued path.

The library allows for various custom versions of maps, as the card and board basis is read in dynamically. 

**Be careful** as the state abbreviation NE is found twice on the original map (Nebraska and New England). For clarity, New England is abbreviated as NE2 as state as well as the card symbol.
### Example Usage
In the [example folder](https://github.com/P-Muench/OnTouR/tree/main/examples) are two examples supplied. 
1. The script `opt_random_game.py` can produce a random round of OnTour and hand it to the MIP. Various scenarios can be gernerated via different seeds and boards.
2. In `opt_given_game.py` there is an implementation of how a given game (found in `given_game.txt`) can be read in and optimized. Only supply the drawn cards and rolled dices per round. 

### Dynamic boards
The standard USA map of OnTour is given in the [resources folder](https://github.com/P-Muench/OnTouR/tree/main/resources) as `board_orig.txt` and `cards_orig.txt`. Custom maps can be added in the same format. There is furthermore an option to reduce the original map to a subset via `reduce_orig.py` that, given a subset of states, produces the according map and board info. 