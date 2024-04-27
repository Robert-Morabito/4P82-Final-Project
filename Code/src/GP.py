import numpy as np
from random import random
from deap import algorithms, base, creator, tools, gp


class wordPredictGP():
    def __init__(self, train_data, test_data, params):
        self.train_data = train_data
        self.test_data = test_data
        self.cx = params['cx']
        self.mut = params['mut']
        self.pop = params['pop']
        self.maxfiteval = params['maxfiteval']
        self.elit = params['elit']
        self.fiteval = 0
        self.pset = None
        self.toolbox = None
        self.setup_pset()
        self.setup_toolbox()

    def vec_add(self, u, v):
        # Get result, get norm of result, normalize vector to be unit length
        result = u + v
        norm = np.linalg.norm(result)
        return result / norm if norm != 0 else result

    def vec_sub(self, u, v):
        # Get result, get norm of result, normalize vector to be unit length
        result = u - v
        norm = np.linalg.norm(result)
        return result / norm if norm != 0 else result

    def vec_mul(self, u, v):
        # Get result, get norm of result, normalize vector to be unit length
        result = u * v
        norm = np.linalg.norm(result)
        return result / norm if norm != 0 else result

    def vec_prot_div(self, u, v):
        # Get result, get norm of result, normalize vector to be unit length
        result = u + v
        norm = np.linalg.norm(result)
        return result / norm if norm != 0 else result

    def vec_sqr(self, u, v):
        # Get result, get norm of result, normalize vector to be unit length
        result = u + v
        norm = np.linalg.norm(result)
        return result / norm if norm != 0 else result

    def vec_root(self, u, v):
        # Get result, get norm of result, normalize vector to be unit length
        result = u + v
        norm = np.linalg.norm(result)
        return result / norm if norm != 0 else result

    def cxUniform(self, ind1, ind2, indpb):
        return tools.cxUniform(ind1, ind2, indpb)

    def cxOnePoint(self, ind1, ind2):
        return tools.cxOnePoint(ind1, ind2)

    def cxTwoPoint(self, ind1, ind2):
        return tools.cxTwoPoint(ind1, ind2)

    # This allows us to use multiple crossover methods like the paper, though we cannot do the same types as they did
    # using the built-in methods that come with DEAP
    def random_crossover(self, ind1, ind2):
        crossovers = [(self.cxUniform, {"indpb": 0.1}),
                      (self.cxOnePoint, {}),
                      (self.cxTwoPoint, {})]
        cx, args = random.choice(crossovers)
        return cx(ind1, ind2, **args)

    def fitness(self, individual):
        # TODO: Define fitness function
        self.fiteval += 1 # Increment the total evaluations done
        return sum(individual)  # Temp placeholder

    def setup_pset(self):
        # TODO: Define pset functions
        a = 1

    def setup_toolbox(self):
        # Initialize creator
        creator.create('FitnessMax', base.Fitness, weights=(1.0,))
        creator.create('Individual', gp.PrimitiveTree, fitness=creator.FitnessMax)

        # Initialize toolbox
        self.toolbox = base.Toolbox()
        self.toolbox.register("expr_init", gp.genHalfAndHalf, pset=self.pset, min_=1, max_=5)
        self.toolbox.register("individual", tools.initIterate, creator.Individual, self.toolbox.expr_init)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        self.toolbox.register("compile", gp.compile, pset=self.pset)
        self.toolbox.register("evaluate", self.fitness)
        self.toolbox.register("select", tools.selTournament, tournsize=3)
        self.toolbox.register("mate", self.random_crossover)
        self.toolbox.register("expr_mut", gp.genFull, min_=1, max_=2)
        self.toolbox.register("mutate", gp.mutUniform, expr=self.toolbox.expr_mut, pset=self.pset)

    def train_GP(self):
        # Initialization
        pop = self.toolbox.population(self.pop)
        hof = tools.HallOfFame(self.elit)
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("std", np.std)
        stats.register("min", np.min)
        stats.register("max", np.max)

        # Train GP system using what was described in the paper
        while self.fiteval < self.maxfiteval:
            for i in range(self.pop):
                # Perform tournament selection and sort by fitness values
                tournament = self.toolbox.select(pop, 3)
                tournament.sort(key=lambda ind: ind.fitness.values, reverse=True)

                # Mate the top 2 and get the best child
                ch1, ch2 = self.toolbox.mate(tournament[0], tournament[1])
                child = ch1 if ch1.fitness >= ch2.fitness else ch2

                # Mutate and evluate result
                self.toolbox.mutate(child)
                child.fitness.values = self.toolbox.evaluate(child)

                # Replace worst in the tournament
                pop.pop(pop.index(tournament[-1]))
                pop.append(child)

                # Incase we exceed during the loop
                if self.fiteval >= self.maxfiteval:
                    break

        return pop, stats, hof
