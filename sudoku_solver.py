import numpy as np


class Solver():

    def __init__(self, sudoku):
        self.sudoku = sudoku
        self.stop = False

    def is_valid(self, y, x, n):
        if n in self.sudoku[:, x] or n in self.sudoku[y, :]:
            return False

        sq_x = (x//3)*3
        sq_y = (y//3)*3

        if n in self.sudoku[sq_y:sq_y+3, sq_x:sq_x+3]:
            return False

        return True

    def find_empty(self):
        for i in range(len(self.sudoku)):
            for j in range(len(self.sudoku[0])):
                if self.sudoku[i][j] == 0:
                    return (i, j)
        return None

    # def solve(self):
    #     coord = self.find_empty()
    #     if coord is None:
    #         return True
    #     x, y = coord

    #     for n in range(1, 10):
    #         if self.is_valid(y, x, n):
    #             self.sudoku[y][x] = n
    #             if self.solve():
    #                 return self.sudoku
    #             self.sudoku[y][x] = 0
    #     return False

    def solve(self):
        find = self.find_empty()
        if not find:
            return True
        else:
            row, col = find

        for i in range(1, 10):
            if self.is_valid(row, col, i):
                self.sudoku[row][col] = i

                if self.solve():
                    return True

                self.sudoku[row][col] = 0

        return False
