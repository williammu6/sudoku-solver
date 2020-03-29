import numpy as np


class Solver():

    def __init__(self, sudoku):
        self.sudoku = sudoku

    def is_valid(self, y, x, n):
        if n in self.sudoku[:, x] or n in self.sudoku[y, :]:
            return False

        sq_x = (x//3)*3
        sq_y = (y//3)*3

        if n in self.sudoku[sq_y:sq_y+3, sq_x:sq_x+3]:
            return False

        return True

    def find_empty(self):
        for x in range(len(self.sudoku)):
            for y in range(len(self.sudoku[0])):
                if self.sudoku[x][y] == 0:
                    return (x, y)
        return None

    def solve(self):
        coord = self.find_empty()
        if not coord:
            return True

        x, y = coord

        for i in range(1, 10):
            if self.is_valid(x, y, i):
                self.sudoku[x][y] = i

                if self.solve():
                    return True

                self.sudoku[x][y] = 0

        return False
