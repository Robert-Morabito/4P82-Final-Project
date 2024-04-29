import numpy as np
from deap import algorithms, base, creator, tools, gp
from scipy.spatial import distance
from vector_operators import vec_add, vec_sub, vec_mul, vec_prot_div, vec_sqr, vec_root


class WordPredictGP:
    def __init__(self, train_data, test_data, params, toolbox):
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
        self.pset = None
        self.toolbox = toolbox if toolbox is not None else base.Toolbox()
        self.setup_pset()
        self.setup_toolbox()

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
            inputs = entry[:-1]  # Prediction from first 5 words
            target = entry[-1]  # Target 6th word
            predicted = func(*inputs)

            if np.linalg.norm(predicted) == 0 or np.linalg.norm(target) == 0:  # Avoid div/0 error in cosine sim
                similarity = 0
            else:
                similarity = 1 - distance.cosine(predicted, target)
            similarities.append(similarity)
        # Return the average similarity across the dataset
        return (np.mean(similarities),)

    def testing(self, individual, dataset):
        """
        Calculates the fitness of the individual by finding cosine similarity between predicted and target vectors
        :param individual: Current tree
        :param dataset: Training dataset
        :return: Returns cosine similarity between predicted and target
        """
        func = self.toolbox.compile(expr=individual)
        vecs = []
        i = 0
        similarities = []
        for entry in dataset:
            inputs = entry[:-1]  # Prediction from first 5 words
            target = entry[-1]  # Target 6th word
            predicted = func(*inputs)

            # Tracks a set number of outputs for the qualitative examples
            if i < 10:
                vecs.append(predicted)
                i += 1

            if np.linalg.norm(predicted) == 0 or np.linalg.norm(target) == 0:  # Avoid div/0 error in cosine sim
                similarity = 0
            else:
                similarity = 1 - distance.cosine(predicted, target)
            similarities.append(similarity)
        # Return the average similarity across the dataset
        return vecs, (np.mean(similarities),)

    def setup_pset(self):
        """
        Initializes the operators and terminals for the primitive set
        """
        # Defining data types as arrays and not the default scalar inputs
        input_type = np.ndarray
        output_type = np.ndarray

        # Initialize primitive set
        self.pset = gp.PrimitiveSetTyped("MAIN", [input_type, input_type, input_type, input_type, input_type],
                                         output_type)

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
        self.toolbox.register("expr_init", gp.genHalfAndHalf, pset=self.pset, min_=1, max_=3)
        self.toolbox.register("individual", tools.initIterate, creator.Individual, self.toolbox.expr_init)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        self.toolbox.register("compile", gp.compile, pset=self.pset)
        self.toolbox.register("evaluate", self.fitness, dataset=self.train_data)
        self.toolbox.register("select", tools.selTournament, tournsize=3)
        self.toolbox.register("mate", gp.cxOnePoint)
        self.toolbox.register("expr_mut", gp.genFull, min_=1, max_=2)
        self.toolbox.register("mutate", gp.mutUniform, expr=self.toolbox.expr_mut, pset=self.pset)

    def run_gp(self):
        """
        Evolves a GP model designed to perform word prediction
        :return:
        """
        # Initialization
        pop = self.toolbox.population(n=self.pop)
        hof = tools.HallOfFame(self.elit)
        stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
        stats_size = tools.Statistics(len)
        stats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
        stats.register("avg", np.mean)
        stats.register("std", np.std)
        stats.register("min", np.min)
        stats.register("max", np.max)

        pop, logs = algorithms.eaSimple(pop, self.toolbox, cxpb=self.cx, mutpb=self.mut, ngen=self.gens,
                                        stats=stats, verbose=True, halloffame=hof)

        # Run the testing set
        vecs, test_fit = self.testing(hof[0], self.test_data)

        return logs, vecs, test_fit[0]
