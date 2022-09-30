import random
import time

from matplotlib import pyplot as plt

from Graph_Creator import *

'''Genetic Algorithm
Made by Pranav Ruparel -> Sep 2022'''

# These are the hyper-parameters used in the code

# Number of Vertices and Edges in the graph
V = 50
E = [100]

# number of iterations of the genetic algorithm to be performed
ITERATIONS_PER_SIZE = 6

# Number of members that are present in each generation
GENERATION_SIZE = 80

# Number of generations that the algorithm will run for
GENERATIONS = 50

# small probability that will decide whether a mutation will occur in an individual or not
MUTATE_PROBABILITY = 0.1

# mixing number, the number of parents considered for making a child
RHO = 5

# i.e take the top # fittest states into the next generation
ELITIST_NUMBER = 15

# discard the bottom # most unfit states from taking part in reproduction
CULLING_NUMBER = 15


# Coloring class is the template that defines how each member of our population is built.
class Coloring:
    # colors dict to convert numbers to colors
    colors = {0: "R", 1: "G", 2: "B"}

    # initialize a new Coloring object given its state and find its fitness
    def __init__(self, state):
        self.state = state
        self.fitness = self.calculate_fitness()

    # function to calculate the fitness of a coloring object
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

        # return the answer
        return fitness_cnt

    # string formatting to print the coloring object and its state
    def __str__(self):
        out = ""  # f"Fitness: {self.fitness} \n"
        for i in range(V):
            out += str(i + 1) + ":" + Coloring.colors[self.state[i]] + ", "

        out = out.rstrip(", ")
        return out


# function to generate a random coloring for V vertices
def generate_state():
    return np.random.randint(0, 3, V)


# reproduce by mixing the Colorings of RHO parents
def reproduce(parents, prob=MUTATE_PROBABILITY * 1.0, use_better_mutation=False):
    # container for child state
    child_state = np.array([])

    # Update 8:
    # randomly shuffle the parents list, to increase randomization
    random.shuffle(parents)

    # to make a child state from RHO parents RHO - 1 random indexes should be chosen as indices for partitions
    # r stores the indices of the partitions, (0, 50) included for programming ease
    r = np.random.randint(0, high=V, size=(1, RHO - 1), dtype=int)
    r = np.append(r, [0, V])
    r.sort()

    # iterate through partitions
    parent_ptr = 0
    for i in range(RHO):
        # p1 indicates the start of the copying region, and p2 the end
        p1 = r[i]
        p2 = r[i + 1]

        # add the selected genes to child
        child_state = np.append(child_state, parents[parent_ptr].state[p1:p2])
        # go to the next parent
        parent_ptr += 1

    # call mutation function on the child state, based on call
    if use_better_mutation:
        child_state = mutate_if_better(child_state, prob)
    else:
        child_state = mutate(child_state, prob)

    child_state = np.array([int(x) for x in child_state])

    return Coloring(child_state)


# randomly mutate the child state, based on the MUTATE_PROBABILITY
def mutate(child, prob=MUTATE_PROBABILITY):
    # here child is a list of colors for each vertex
    r = np.random.randint(1, 101)

    # if the random case occurs
    if r <= prob * 100:
        # choose an index to mutate
        idx = np.random.randint(0, V)
        # choose the color to change
        child[idx] = np.random.randint(0, 3)

    return child


# accept a mutation only if it has a better fitness as compared to the original
def mutate_if_better(child, prob):
    temp = child

    # here child is a list of colors for each vertex
    r = np.random.randint(1, 101)

    # if the random case occurs
    if r <= prob * 100:
        # choose an index to mutate
        idx = np.random.randint(0, V)
        # choose the color to change
        child[idx] = np.random.randint(0, 3)

    if Coloring(temp).fitness < Coloring(child).fitness:
        return child

    return temp


# check if cumulative sums are zero
def cumulative_zero(weights):
    s = np.sum(np.array(weights))
    return s == 0


# function to generate population
def generate_population(p=GENERATION_SIZE):
    members = []
    f_members = []

    # add the states to generation
    for i in range(p):
        s = Coloring(generate_state())
        members.append(s)
        f_members.append(s.fitness)

    return members, f_members


# function to generate global graph, given a graph creator object and a edge_length
def generate_random_graph(gc, edge_length):
    edges = gc.CreateGraphWithRandomEdges(edge_length)  # Creates a random graph with 50 vertices and 100 edges

    global g
    g = np.zeros((V, V), dtype=int)

    for edge in edges:
        g[edge[0]][edge[1]] = 1
        g[edge[1]][edge[0]] = 1


# function to make a graph, given the testcase file name
def generate_test_graph(gc, filename):
    edges = gc.ReadGraphfromCSVfile(filename)

    global g
    g = np.zeros((V, V), dtype=int)

    for edge in edges:
        g[edge[0]][edge[1]] = 1
        g[edge[1]][edge[0]] = 1

    return int(filename[-3:])


def main():
    # create an object of the graph creator class
    gc = Graph_Creator(V)

    # generate_random_graph(gc, edge_length)
    edge_length = generate_test_graph(gc, "Testcases/100")

    # containers to find and store the best state
    fitness_of_best_child = -1
    max_fitness_of_generation = []
    best_fit_child = None

    # start time to find running time
    start = time.time()

    # Run the Genetic Algorithm Several times
    for _ in range(ITERATIONS_PER_SIZE):

        # make the first generation
        generation, fitness_matrix = generate_population()

        # generation 0 is the initial population
        max_fitness, best_child = -1, 0
        fitness_array = []

        # start iterating through the generations
        for _ in range(1, GENERATIONS):

            # containers to contain new generation
            new_generation, new_fitness_matrix = [], []

            # update 5: Elitism
            # sort the current generation and pick the best ones
            generation.sort(key=lambda a: a.fitness, reverse=True)
            for k in range(ELITIST_NUMBER):
                new_generation.append(generation[k])
                new_fitness_matrix.append(generation[k].fitness)

            # if the current fitness matrix isn't all zeroes
            if not cumulative_zero(fitness_matrix):
                # generate V - ELITIST_NUMBER number of children
                for _ in range(GENERATION_SIZE - ELITIST_NUMBER):

                    # blank array that will contain RHO number of parents
                    parents = []
                    try:
                        # add the parents based on weights defined as fitness
                        for k in range(RHO):
                            parents.append(random.choices(generation, weights=fitness_matrix, k=1)[0])

                    except ValueError as ve:
                        # catch value error
                        print(ve)
                        break

                    # this is the result of sexual reproduction along with mutation in a small probability
                    # update 2: using the better mutation function.
                    child = reproduce(parents, MUTATE_PROBABILITY, use_better_mutation=True)

                    new_generation.append(child)
                    new_fitness_matrix.append(child.fitness)

                    if child.fitness > max_fitness:
                        best_child = child
                        max_fitness = child.fitness
            else:
                # Update 4: randomly generate a new generation, if previous generations don't have enough fit
                # parents space here for random restart
                new_generation, new_fitness_matrix = generate_population()

            # update 6: implement Culling
            # discard the bottom # most unfit states from taking part in reproduction
            generation.sort(key=lambda a: a.fitness, reverse=True)
            new_fitness_matrix.sort(reverse=True)
            for _ in range(CULLING_NUMBER):
                new_generation.pop()
                new_fitness_matrix.pop()

            # update the generation and fitness for the next generation
            generation = new_generation
            fitness_matrix = new_fitness_matrix

            # this array is used for plotting the graph
            fitness_array.append(max_fitness)

        # update the answers for a given edge size
        if max_fitness > fitness_of_best_child:
            fitness_of_best_child = max_fitness
            best_fit_child = best_child
            max_fitness_of_generation = fitness_array

    # end time to now calculate the final running time
    end = time.time()

    # print the final answer
    print("Roll no : 2020A7PS0973G")
    print("Number of Edges : " + str(edge_length))
    print("Best State : ")
    print(best_fit_child)
    print("Fitness Value of Best State : " + str(best_fit_child.fitness))
    print("Time taken : " + str(round(end - start, 2)) + " seconds")

    # plot how the max fitness changes as the generations pass by
    plt.plot(max_fitness_of_generation, color='c', label=f'Edges: {edge_length}')
    plt.legend(loc='best')
    plt.title("Max Fitness vs Generations")
    plt.ylim([0, GENERATIONS])
    plt.xlabel("Generations")
    plt.ylabel("Fitness (50)")
    plt.show()


if __name__ == '__main__':
    main()
