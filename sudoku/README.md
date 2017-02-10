# Artificial Intelligence Nanodegree
## Introductory Project: Diagonal Sudoku Solver

# Question 1 (Naked Twins)
Q: How do we use constraint propagation to solve the naked twins problem?
A: Constraint propagation was used in order to reduce the search space of the problem, by using the naked twins technique. Once it is applied, a new sub-tree in the search space is created and traversed until a solution is found.

This technique consists in the following: Given two unsolved boxes that belong to a same unit, each with only the same 2 digits as possibilities, no other box may take the value of any of those digits. If it did, then one of the two 'twins' wouldn't have a value to take and the board would become invalid.


# Question 2 (Diagonal Sudoku)
Q: How do we use constraint propagation to solve the diagonal sudoku problem?
A: The diagonal sudoku problem is solved by adding the constraint that no boxes in a same diagonal can take already used digits. It is implemented by creating two new (box) units that contain the two diagonals of the sudoku, and taking them into account during tree prunning (during Deletion or Naked Twins, for this project).

Just as with the previous question, constraint propagation is used in order to reduce the search state, instead of trying all different possible sudoku board configurations until reaching to the correct one. A Sudoku game has at most 9^81 different games(valid & invalid scenarios), which is a number very big to even try, so constraint propagation allows us to use constraints (diagonal sudoku) in order to reduce the possible scenarios before trying them.

### Install

This project requires **Python 3**.

We recommend students install [Anaconda](https://www.continuum.io/downloads), a pre-packaged Python distribution that contains all of the necessary libraries and software for this project.
Please try using the environment we provided in the Anaconda lesson of the Nanodegree.

##### Optional: Pygame

Optionally, you can also install pygame if you want to see your visualization. If you've followed our instructions for setting up our conda environment, you should be all set.

If not, please see how to download pygame [here](http://www.pygame.org/download.shtml).

### Code

* `solutions.py` - You'll fill this in as part of your solution.
* `solution_test.py` - Do not modify this. You can test your solution by running `python solution_test.py`.
* `PySudoku.py` - Do not modify this. This is code for visualizing your solution.
* `visualize.py` - Do not modify this. This is code for visualizing your solution.

### Visualizing

To visualize your solution, please only assign values to the values_dict using the ```assign_values``` function provided in function.py

### Data

The data consists of a text file of diagonal sudokus for you to solve.
