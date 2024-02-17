import random
import time

import matplotlib.pyplot as plt


class WilsonMazeGenerator:
    """
    Maze Generator using Wilson's Loop Erased Random Walk Algorithm

    Source: https://github.com/CaptainFl1nt/WilsonMazeGenerator
    """

    def __init__(self, height, width):
        """
        WilsonMazeGenerator(int,int) -> WilsonMazeGenerator
        Creates a maze generator with specified width and height.
        width: width of generated mazes
        height: height of generated mazes
        """
        self.width = 2 * (width // 2) + 1  # Make width odd
        self.height = 2 * (height // 2) + 1  # Make height odd

        # grid of cells
        self.grid = [[0 for j in range(self.width)] for i in range(self.height)]

        # declare instance variable
        self.visited = []  # visited cells
        self.unvisited = []  # unvisited cells
        self.path = dict()  # random walk path

        # valid directions in random walk
        self.directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]

        # indicates whether a maze is generated
        self.generated = False

        # shortest solution
        self.solution = []
        self.showSolution = False
        self.start = (0, 0)
        self.end = (self.height - 1, self.width - 1)

    def __str__(self):
        """
        WilsonMazeGenerator.__str__() -> str
        outputs a string version of the grid
        """
        out = "##" * (self.width + 1) + "\n"
        for i in range(self.height):
            out += "#"
            for j in range(self.width):
                if self.grid[i][j] == 0:
                    out += "##"
                else:
                    if not self.showSolution:
                        out += "  "
                    elif (i, j) in self.solution:
                        out += "**"
                    else:
                        out += "  "
            out += "#\n"
        return out + "##" * (self.width + 1)

    def get_grid(self):
        """WilsonMazeGenerator.get_grid() -> list
        returns the maze grid"""
        return self.grid

    def get_solution(self):
        """WilsonMazeGenerator.get_solution() -> list
        Returns the solution to the maze as a list
        of tuples"""
        return self.solution

    def show_solution(self, show):
        """WilsonMazeGenerator.show_solution(boolean) -> None
        Set whether WilsonMazeGenerator.__str__() outputs the
        solution or not"""
        self.showSolution = show

    def generate_maze(self):
        """WilsonMazeGenerator.generate_maze() -> None
        Generates the maze according to the Wilson Loop Erased Random
        Walk Algorithm"""
        # reset the grid before generation
        self.initialize_grid()

        # choose the first cell to put in the visited list
        # see Step 1 of the algorithm.
        current = self.unvisited.pop(random.randint(0, len(self.unvisited) - 1))
        self.visited.append(current)
        self.cut(current)

        # loop until all cells have been visited
        while len(self.unvisited) > 0:
            # choose a random cell to start the walk (Step 2)
            first = self.unvisited[random.randint(0, len(self.unvisited) - 1)]
            current = first
            # loop until the random walk reaches a visited cell
            while True:
                # choose direction to walk (Step 3)
                dirNum = random.randint(0, 3)
                # check if direction is valid. If not, choose new direction
                while not self.is_valid_direction(current, dirNum):
                    dirNum = random.randint(0, 3)
                # save the cell and direction in the path
                self.path[current] = dirNum
                # get the next cell in that direction
                current = self.get_next_cell(current, dirNum, 2)
                if current in self.visited:  # visited cell is reached (Step 5)
                    break

            current = first  # go to start of path
            # loop until the end of path is reached
            while True:
                # add cell to visited and cut into the maze
                self.visited.append(current)
                self.unvisited.remove(current)  # (Step 6.b)
                self.cut(current)

                # follow the direction to next cell (Step 6.a)
                dirNum = self.path[current]
                crossed = self.get_next_cell(current, dirNum, 1)
                self.cut(crossed)  # cut crossed edge

                current = self.get_next_cell(current, dirNum, 2)
                if current in self.visited:  # end of path is reached
                    self.path = dict()  # clear the path
                    break

        self.generated = True

    def solve_maze(self):
        """WilsonMazeGenerator.solve_maze() -> None
        Solves the maze according to the Wilson Loop Erased Random
        Walk Algorithm"""
        # if there is no maze to solve, cut the method
        if not self.generated:
            return None

        # initialize with empty path at starting cell
        self.path = dict()
        current = self.start

        # loop until the ending cell is reached
        while True:
            while True:
                # choose valid direction
                # must remain in the grid
                # also must not cross a wall
                dirNum = random.randint(0, 3)
                adjacent = self.get_next_cell(current, dirNum, 1)
                if self.is_valid_direction(current, dirNum):
                    hasWall = self.grid[adjacent[0]][adjacent[1]] == 0
                    if not hasWall:
                        break
            # add cell and direction to path
            self.path[current] = dirNum

            # get next cell
            current = self.get_next_cell(current, dirNum, 2)
            if current == self.end:
                break  # break if ending cell is reached

        # go to start of path
        current = self.start
        self.solution.append(current)
        # loop until end of path is reached
        while not (current == self.end):
            dirNum = self.path[current]  # get direction
            # add adjacent and crossed cells to solution
            crossed = self.get_next_cell(current, dirNum, 1)
            current = self.get_next_cell(current, dirNum, 2)
            self.solution.append(crossed)
            self.solution.append(current)

        self.path = dict()

    ## Private Methods ##
    ## Do Not Use Outside This Class ##

    def get_next_cell(self, cell, dirNum, fact):
        """WilsonMazeGenerator.get_next_cell(tuple,int,int) -> tuple
        Outputs the next cell when moved a distance fact in the the
        direction specified by dirNum from the initial cell.
        cell: tuple (y,x) representing position of initial cell
        dirNum: int with values 0,1,2,3
        fact: int distance to next cell"""
        dirTup = self.directions[dirNum]
        return (cell[0] + fact * dirTup[0], cell[1] + fact * dirTup[1])

    def is_valid_direction(self, cell, dirNum):
        """WilsonMazeGenerator(tuple,int) -> boolean
        Checks if the adjacent cell in the direction specified by
        dirNum is within the grid
        cell: tuple (y,x) representing position of initial cell
        dirNum: int with values 0,1,2,3"""
        newCell = self.get_next_cell(cell, dirNum, 2)
        tooSmall = newCell[0] < 0 or newCell[1] < 0
        tooBig = newCell[0] >= self.height or newCell[1] >= self.width
        return not (tooSmall or tooBig)

    def initialize_grid(self):
        """
        WilsonMazeGenerator.initialize_grid() -> None
        Resets the maze grid to blank before generating a maze.
        """
        for i in range(self.height):
            for j in range(self.width):
                self.grid[i][j] = 0

        # fill up unvisited cells
        for r in range(self.height):
            for c in range(self.width):
                if r % 2 == 0 and c % 2 == 0:
                    self.unvisited.append((r, c))

        self.visited = []
        self.path = dict()
        self.generated = False

    def cut(self, cell):
        """WilsonMazeGenerator.cut(tuple) -> None
        Sets the value of the grid at the location specified by cell
        to 1
        cell: tuple (y,x) location of where to cut"""
        self.grid[cell[0]][cell[1]] = 1


if __name__ == "__main__":

    ##########################################################
    ## Adjust maze size here! Width and Height must be odd! ##
    ##########################################################
    width = 9  ##
    height = 9  ##
    ##########################################################

    fact = 0.06666667  # size of cell
    # width * fact is width in inches
    # height * fact is height in inches

    generator = WilsonMazeGenerator(height, width)
    generator.generate_maze()
    print("Maze Generated")

    fig, ax = plt.subplots()
    fig.tight_layout(pad=0.0)

    grid = generator.get_grid()

    # draw border
    ax.vlines(-1, -1, -height, linewidth=1)
    ax.vlines(width, 1, -height + 2, linewidth=1)
    ax.hlines(1, -1, width, linewidth=1)
    ax.hlines(-height, -1, width, linewidth=1)

    # draw maze
    for i in range(height):
        for j in range(width - 1):
            if grid[i][j] == 0 and grid[i][j + 1] == 0:
                ax.hlines(-i, j, j + 1, linewidth=1)

    for i in range(width):
        for j in range(height - 1):
            if grid[j][i] == 0 and grid[j + 1][i] == 0:
                ax.vlines(i, -j, -(j + 1), linewidth=1)

    for j in range(width):
        if grid[0][j] == 0:
            ax.vlines(j, 1, 0, linewidth=1)
        if grid[height - 1][j] == 0:
            ax.vlines(j, -height, -height + 1, linewidth=1)
    for j in range(height):
        if grid[j][0] == 0:
            ax.hlines(-j, -1, 0, linewidth=1)
        if grid[j][width - 1] == 0:
            ax.hlines(-j, width, width - 1, linewidth=1)

    print("Maze Drawn")

    fig.set_size_inches(width * fact, height * fact)
    plt.axis("off")
    idn = str(int(time.time()))
    # plt.savefig("maze"+idn+".png",bbox_inches="tight",transparent=True)
    print("Maze Saved as maze" + idn + ".png")

    # draw solution
    generator.solve_maze()
    soln = generator.get_solution()
    print("Solution Generated")
    ax.hlines(0, -1, 0, linewidth=1, colors="red")
    ax.hlines(-height + 1, width - 1, width, linewidth=1, colors="red")
    for i in range(len(soln) - 1):
        if soln[i][0] == soln[i + 1][0]:
            ax.hlines(
                -soln[i][0], soln[i][1], soln[i + 1][1], linewidth=1, colors="red"
            )
        else:
            ax.vlines(
                soln[i][1], -soln[i][0], -soln[i + 1][0], linewidth=1, colors="red"
            )
    print("Solution Drawn")
    # plt.savefig("maze"+idn+"_answer.png",bbox_inches="tight",transparent=True)
    print("Solution Saved as maze" + idn + "_answer.png")
    plt.show()
