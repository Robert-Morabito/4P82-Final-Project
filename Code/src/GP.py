import numpy as np
from random import random, choice
from deap import algorithms, base, creator, tools, gp
from scipy.spatial import distance


def vec_add(u, v):
    """
    Adds two vectors and normalizes them
    :param u: First vector
    :param v: Second vector
    :return: Returns vector after adding
    """
    result = u + v
    norm = np.linalg.norm(result)
    return result / norm if norm != 0 else result


def vec_sub(u, v):
    """
    Subtracts two vectors and normalizes them
    :param u: First vector
    :param v: Second vector
    :return: Returns vector after subtracting
    """
    result = u - v
    norm = np.linalg.norm(result)
    return result / norm if norm != 0 else result


def vec_mul(u, v):
    """
    Multiplies two vectors and normalizes them
    :param u: First vector
    :param v: Second vector
    :return: Returns vector after multiplying
    """
    result = u * v
    norm = np.linalg.norm(result)
    return result / norm if norm != 0 else result


def vec_prot_div(u, v):
    """
    Divides two vectors and normalizes them (protects against div by 0)
    :param u: First vector
    :param v: Second vector
    :return: Returns vector after dividing
    """
    v = np.where(v == 0, 1, v) # Ensures den will be 1 if ever 0
    result = u / v
    norm = np.linalg.norm(result)
    return result / norm if norm != 0 else result


def vec_sqr(u):
    """
    Calculates the square of a vector
    :param u: Vector
    :return: Returns vector after squaring
    """
    result = u ** 2
    norm = np.linalg.norm(result)
    return result / norm if norm != 0 else result


def vec_root(u):
    """
    Calculates the root of a vector
    :param u: Vector
    :return: Returns vector after square root
    """
    result = np.sqrt(np.abs(u))
    norm = np.linalg.norm(result)
    return result / norm if norm != 0 else result


class WordPredictGP:
    def __init__(self, train_data, test_data, params):
        """
        Initializes the data, parameters, primitive set, and toolbox for a GP system
        :param train_data: Training data
        :param test_data: Testing data
        :param params: GP parameters
        """
        self.train_data = train_data
        self.test_data = test_data
        self.cx = params['cx']
        self.mut = params['mut']
        self.pop = params['pop']
        self.gens = params['gens']
        self.elit = params['elit']
        #self.fiteval = 0
        self.pset = None
        self.toolbox = None
        self.setup_pset()
        self.setup_toolbox()

    # def cxUniform(self, ind1, ind2, indpb):
    #     """
    #     Uniform crossover operator
    #     :param ind1: First individual
    #     :param ind2: Second individual
    #     :param indpb: Individual probability
    #     :return: Result of crossover
    #     """
    #     return tools.cxUniform(ind1, ind2, indpb)
    #
    # def cxOnePoint(self, ind1, ind2):
    #     """
    #     One point crossover operator
    #     :param ind1: First individual
    #     :param ind2: Second individual
    #     :return: Result of crossover
    #     """
    #     return tools.cxOnePoint(ind1, ind2)
    #
    # def cxTwoPoint(self, ind1, ind2):
    #     """
    #     Two point crossover operator
    #     :param ind1: First individual
    #     :param ind2: Second individual
    #     :return: Result of crossover
    #     """
    #     return tools.cxTwoPoint(ind1, ind2)
    #
    # # This allows us to use multiple crossover methods like the paper, though we cannot do the same types as they did
    # # using the built-in methods that come with DEAP
    # def random_crossover(self, ind1, ind2):
    #     """
    #     Performs a random crossover from uniform, one point and two points.
    #     :param ind1: First individual
    #     :param ind2: Second individual
    #     :return: Result of crossover
    #     """
    #     crossovers = [(self.cxUniform, {"indpb": 0.1}),
    #                   (self.cxOnePoint, {}),
    #                   (self.cxTwoPoint, {})]
    #     cx, args = choice(crossovers)
    #     return cx(ind1, ind2, **args)

    def fitness(self, individual, dataset):
        """
        Calculates the fitness of the individual by finding cosine similarity between predicted and target vectors
        :param individual: Current tree
        :param dataset: Training dataset
        :return: Returns cosine similarity between predicted and target
        """
        func = self.toolbox.compile(expr=individual)
        similarities = []
        for entry in dataset:
            inputs = entry[:-1]   # Prediction from first 5 words
            target = entry[-1]  # Target 6th word
            predicted = func(*inputs)

            if np.linalg.norm(predicted) == 0 or np.linalg.norm(target) == 0:   # Avoid div/0 error in cosine sim
                similarity = 0
            else:
                similarity = 1 - distance.cosine(predicted, target)
            similarities.append(similarity)
        # Return the average similarity across the dataset
        return (np.mean(similarities),)

    def setup_pset(self):
        """
        Initializes the operators and terminals for the primitive set
        """
        # Defining data types as arrays and not the default scalar inputs
        input_type = np.ndarray
        output_type = np.ndarray

        # Initialize primitive set
        self.pset = gp.PrimitiveSetTyped("MAIN", [input_type, input_type, input_type, input_type, input_type], output_type)

        # Binary operators
        self.pset.addPrimitive(vec_add, [input_type, input_type], output_type)
        self.pset.addPrimitive(vec_sub, [input_type, input_type], output_type)
        self.pset.addPrimitive(vec_mul, [input_type, input_type], output_type)
        self.pset.addPrimitive(vec_prot_div, [input_type, input_type], output_type)

        # Unary operators
        self.pset.addPrimitive(vec_sqr, [input_type], output_type)
        self.pset.addPrimitive(vec_root, [input_type], output_type)

        # Renaming arguments for terminals
        self.pset.renameArguments(ARG0='w0', ARG1='w1', ARG2='w2', ARG3='w3', ARG4='w4')

    def setup_toolbox(self):
        """
        Initializes the GP toolbox
        """
        # Initialize creator
        creator.create('FitnessMax', base.Fitness, weights=(1.0,))
        creator.create('Individual', gp.PrimitiveTree, fitness=creator.FitnessMax)

        # Initialize toolbox
        self.toolbox = base.Toolbox()
        self.toolbox.register("expr_init", gp.genHalfAndHalf, pset=self.pset, min_=1, max_=5)
        self.toolbox.register("individual", tools.initIterate, creator.Individual, self.toolbox.expr_init)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        self.toolbox.register("compile", gp.compile, pset=self.pset)
        self.toolbox.register("evaluate", self.fitness, dataset=self.train_data)
        self.toolbox.register("select", tools.selTournament, tournsize=3)
        self.toolbox.register("mate", gp.cxOnePoint)
        self.toolbox.register("expr_mut", gp.genFull, min_=1, max_=2)
        self.toolbox.register("mutate", gp.mutUniform, expr=self.toolbox.expr_mut, pset=self.pset)

    def train_gp(self):
        """
        Evolves a GP model designed to perform word prediction
        :return:
        """
        # Initialization
        pop = self.toolbox.population(n=self.pop)
        hof = tools.HallOfFame(self.elit)
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("std", np.std)
        stats.register("min", np.min)
        stats.register("max", np.max)

        pop, logs = algorithms.eaSimple(pop, self.toolbox, cxpb=self.cx, mutpb=self.mut, ngen=self.gens,
                                        stats=stats, verbose=True, halloffame=hof)

        # # Train GP system using what was described in the paper
        # while self.fiteval < self.maxfiteval:
        #     for ind in pop:
        #         entry = choice(self.train_data)
        #         ind.fitness.values = self.toolbox.evaluate(ind, entry)
        #
        #         # Incase we exceed during the loop
        #         if self.fiteval >= self.maxfiteval:
        #             break
        #
        #         # Perform tournament selection and sort by fitness values
        #         tournament = self.toolbox.select(pop, 3)
        #         tournament.sort(key=lambda ind: ind.fitness.values, reverse=True)
        #
        #         # Mate the top 2 and get the best child
        #         ch1, ch2 = self.toolbox.mate(tournament[0], tournament[1])
        #         child = ch1 if ch1.fitness >= ch2.fitness else ch2
        #
        #         # Mutate and evaluate result
        #         self.toolbox.mutate(child)
        #         #child.fitness.values = self.toolbox.evaluate(ind, child)
        #
        #         # Replace worst in the tournament
        #         pop.pop(pop.index(tournament[-1]))
        #         pop.append(child)

        return pop, stats, hof
