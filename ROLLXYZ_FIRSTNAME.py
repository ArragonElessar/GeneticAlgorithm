import math
import random
import time

from matplotlib import pyplot as plt

from Graph_Creator import *

'''Genetic Algorithm
Made by Pranav Ruparel -> Sep 2022'''

V = 50
E = [100]  # set of sizes to generate random edges in the graph
ITERATIONS_PER_SIZE = 7 # 5
GENERATION_SIZE = 80
GENERATIONS = 50
MUTATE_PROBABILITY = 0.1
RHO = 5  # mixing number, the number of parents considered for making a child

ELITIST_NUMBER = 15  # i.e take the top # fittest states into the next generation
CULLING_NUMBER = 15  # discard the bottom # most unfit states from taking part in reproduction


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


# Update 3: Trying to mimic Simulated Annealing, isn't giving good results
# initially, the MUTATION_RATE is set to be very high, and then it gradually reduces
def update_mutation_prob(generation, k=10):
    # new_prob = e ^ (-generation/k)
    return math.pow(math.e, (-generation / k))


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


def generate_test_graph(gc, filename):
    edges = gc.ReadGraphfromCSVfile(filename)

    global g
    g = np.zeros((V, V), dtype=int)

    for edge in edges:
        g[edge[0]][edge[1]] = 1
        g[edge[1]][edge[0]] = 1

    return int(filename[-3:])


def main():
    gc = Graph_Creator(V)

    # generate_random_graph(gc, edge_length)
    edge_length = generate_test_graph(gc, "Testcases/100")

    e_max_fitness = -1
    e_fitness = []
    e_best_child = None

    start = time.time()

    for iteration in range(ITERATIONS_PER_SIZE):

        # make the first generation
        generation, fitness_matrix = generate_population()

        # generation 0 is the initial population
        max_fitness, best_child = -1, 0
        fitness_array = []

        # set the initial mutation probability
        # update 1: set mutate probability according to number of edges
        mutate_probability = 0.1 # edge_length * 0.005

        # start iterating through the generations
        for i in range(1, GENERATIONS):

            # containers to contain new generation
            new_generation, new_fitness_matrix = [], []

            # update 5: Elitism
            # sort the current generation and pick the best ones
            generation.sort(key=lambda a: a.fitness, reverse=True)
            for k in range(ELITIST_NUMBER):
                new_generation.append(generation[k])
                new_fitness_matrix.append(generation[k].fitness)

            # print(f'Number of Members from last generation (Elite): {len(new_generation)}')

            # if the current fitness matrix isn't all zeroes
            if not cumulative_zero(fitness_matrix):
                # generate V - ELITIST_NUMBER number of children
                for j in range(GENERATION_SIZE - ELITIST_NUMBER):

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
                    child = reproduce(parents, mutate_probability, use_better_mutation=True)

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

            # print(f'Number of Members after culling: {len(new_generation)}')

            # update the generation and fitness for the next generation
            generation = new_generation
            fitness_matrix = new_fitness_matrix

            # update the mutation probabilities
            # mutate_probability = update_mutation_prob(i)

            # this array is used for plotting the graph
            fitness_array.append(max_fitness)

        # update the answers for a given edge size
        if max_fitness > e_max_fitness:
            e_max_fitness = max_fitness
            e_best_child = best_child
            e_fitness = fitness_array

    end = time.time()

    print("Roll no : 2020A7PS0973G")
    print("Number of Edges : " + str(edge_length))
    print("Best State : ")
    print(e_best_child)
    print("Fitness Value of Best State : " + str(e_best_child.fitness))
    print("Time taken : " + str(round(end - start, 2)) + " seconds")

    plt.plot(e_fitness, color='c', label=f'Edges: {edge_length}')
    plt.legend(loc='best')
    plt.title("Max Fitness vs Generations")
    plt.ylim([0, GENERATIONS])
    plt.xlabel("Generations")
    plt.ylabel("Fitness (50)")
    plt.show()


if __name__ == '__main__':
    main()
