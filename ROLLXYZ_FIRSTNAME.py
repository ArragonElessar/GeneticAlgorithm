import random
import time

import matplotlib.pyplot as plt

from Graph_Creator import *

'''Genetic Algorithm
Made by Pranav Ruparel -> Sep 2022'''

V = 50
E = [100, 200, 300, 400, 500]  # [100, 200, 300, 400, 500]  # set of sizes to generate random edges in the graph
ITERATIONS_PER_SIZE = 10
INITIAL_POPULATION = 50
GENERATIONS = 50
MUTATE_PROBABILITY = .1

colors = ['c', 'b', 'r', 'y', 'm']


class Coloring:
    colors = {0: "R", 1: "G", 2: "B"}

    def __init__(self, state):
        self.state = state
        self.fitness = self.calculate_fitness()

    def calculate_fitness(self):

        # variable to count number of fit vertices
        fitness_cnt = 0

        # iterate through all the vertices
        for i in range(V):
            # iterate through all the vertices
            isFit = True

            # loop through adjacency matrix, so
            for j in range(V):

                # if there is an edge between vertices (i, j) and both vertices i and j are of the same color
                if i is not j and g[i][j] == 1 and self.state[i] == self.state[j]:
                    # this vertex isn't fir
                    isFit = False
                    break

            # if after checking the color of all neighbouring vertices, a vertex is of different color, it is fir
            if isFit:
                fitness_cnt += 1

        return fitness_cnt

    def __str__(self):
        out = ""  # f"Fitness: {self.fitness} \n"
        for i in range(V):
            out += str(i + 1) + ":" + Coloring.colors[self.state[i]] + ", "

        out = out.rstrip(", ")
        return out


def generate_state():
    return np.random.randint(0, 3, V)


def reproduce(a, b):
    sa = a.state
    sb = b.state

    child_state = [0] * V

    r = np.random.randint(0, V)
    for i in range(V):
        if i <= r:
            child_state[i] = sa[i]
        else:
            child_state[i] = sb[i]

    child_state = mutate(child_state)

    return Coloring(child_state)


def mutate(child):
    # here child is a list of colors for each vertex
    r = np.random.randint(1, 101)
    if r <= MUTATE_PROBABILITY * 100:
        idx = np.random.randint(0, V)
        child[idx] = np.random.randint(0, 3)

    return child


def main():
    gc = Graph_Creator(V)
    color_ptr = 0

    all_size_fitness = []

    for edge_length in E:

        e_max_fitness = -1
        e_fitness = []
        e_best_child = None

        start = time.time()

        for iteration in range(ITERATIONS_PER_SIZE):

            generation = []
            fitness_matrix = []
            edges = gc.CreateGraphWithRandomEdges(edge_length)  # Creates a random graph with 50 vertices and 200 edges

            global g
            g = np.zeros((V, V), dtype=int)

            for edge in edges:
                g[edge[0]][edge[1]] = 1
                g[edge[1]][edge[0]] = 1

            # add the states to first generation
            for i in range(INITIAL_POPULATION):
                state = Coloring(generate_state())
                generation.append(state)
                fitness_matrix.append(state.fitness)

            # generation 0 is the initial population

            max_fitness = -1
            best_child = 0
            fitness_array = []

            # start iterating through the generations
            for i in range(1, GENERATIONS):
                new_generation = []
                new_fitness_matrix = []

                for j in range(V):
                    x = None
                    y = None

                    try:
                        x = random.choices(generation, weights=fitness_matrix, k=1)[0]
                        y = random.choices(generation, weights=fitness_matrix, k=1)[0]

                    except ValueError as ve:
                        print(ve)
                        break

                    # this is the result of sexual reproduction along with mutation in a small probability
                    child = reproduce(x, y)

                    new_generation.append(child)
                    new_fitness_matrix.append(child.fitness)

                    if child.fitness > max_fitness:
                        best_child = child
                        max_fitness = child.fitness

                generation = new_generation
                fitness_matrix = new_fitness_matrix

                # this array is used for plotting the graph
                fitness_array.append(max_fitness)

            if max_fitness > e_max_fitness:
                e_max_fitness = max_fitness
                e_best_child = best_child
                e_fitness = fitness_array

        end = time.time()

        print("Number of Edges: " + str(edge_length))
        print("Best State: ")
        print(e_best_child)
        print("Fitness Value of Best State: " + str(e_best_child.fitness))
        print("Time taken: " + str(round(end - start, 2)) + " seconds")

        plt.plot(e_fitness, color=colors[color_ptr], label=f'Edges: {edge_length}')
        color_ptr += 1
        plt.legend()
        plt.ylim([0, 50])
        plt.xlabel("Generations")
        plt.ylabel("Fitness (50)")
        plt.show()
        all_size_fitness.append(e_fitness)

    color_ptr = 0
    for e_fitness in all_size_fitness:
        plt.plot(e_fitness, color=colors[color_ptr], label=f'Edges: {E[color_ptr]}')
        color_ptr += 1
        plt.legend()
        plt.ylim([0, 50])
        plt.xlabel("Generations")
        plt.ylabel("Fitness (50)")

    plt.show()


if __name__ == '__main__':
    main()
