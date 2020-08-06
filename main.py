import csv
import random

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


class Task:
    def __init__(self, task_array):
        constraints = task_array[0]
        self.W = constraints[1].astype(np.int64)
        self.S = constraints[2].astype(np.int64)
        self.n = constraints[0].astype(np.int64)
        arrays = task_array[1:, :]
        self.w = arrays[:, 0]
        self.s = arrays[:, 1]
        self.c = arrays[:, 2]


class Individual:
    def __init__(self, genotype):
        self.genotype = genotype
        self.genotype_size = genotype.shape[0]

    def evaluate(self, task):
        w_sum = (self.genotype * task.w).sum()
        s_sum = (self.genotype * task.s).sum()
        c_sum = (self.genotype * task.c).sum()
        if w_sum <= task.W and s_sum <= task.S:
            return c_sum
        return 0

    def mutate(self, mutation_rate):
        mutation_size = int(self.genotype_size * mutation_rate)
        mutation = np.random.choice(range(self.genotype_size), mutation_size, replace=False)
        for idx in mutation:
            if self.genotype[idx] == 1:
                self.genotype[idx] = 0
            else:
                self.genotype[idx] = 1


def crossover(crossover_rate, parent_1, parent_2):
    if np.random.random() < crossover_rate:
        split_point = np.random.randint(1, len(parent_1.genotype))
        result_genotype = np.concatenate(
            [parent_1.genotype[:split_point], parent_2.genotype[split_point:]]
        )
        return Individual(result_genotype)
    else:
        return parent_1


class Population:
    def __init__(self, genotype_size=None, pop_size=None):
        self.population = []
        if genotype_size is not None and pop_size is not None:
            population_array = np.random.choice([0, 1], size=(pop_size, genotype_size), p=[0.8, 0.2])
            for genotype in population_array:
                individual = Individual(genotype)
                self.population.append(individual)
        self.size = len(self.population)

    def tournament(self, tournament_size, task):
        selected = random.choices(self.population, k=tournament_size)
        evaluation = [elem.evaluate(task) for elem in selected]
        idx_best_individual = evaluation.index(max(evaluation))
        return selected[idx_best_individual]

    def add_individual(self, individual):
        self.population.append(individual)
        self.size = len(self.population)

    def best(self, task):
        evaluation = []
        for individual in self.population:
            evaluation.append(individual.evaluate(task))

        idx_best_individual = evaluation.index(max(evaluation))
        best = self.population[idx_best_individual]
        return best, best.evaluate(task)

    def evaluate(self, task):
        evaluation = []
        for individual in self.population:
            evaluation.append(individual.evaluate(task))
        mean_evaluation = sum(evaluation) / len(evaluation)
        return mean_evaluation


class GeneticAlgorithm:
    def __init__(self, populations_size, tournament_size, crossover_rate, mutation_rate):
        self.populations_size = populations_size
        self.tournament_size = tournament_size
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate

    def fit(self, iterations, task):
        population = Population(genotype_size=task.n, pop_size=self.populations_size)
        history = []
        fittest_individual, fittest_evaluation = population.best(task)
        for _ in range(iterations):
            new_population = Population()
            for _ in range(population.size):
                parent1 = population.tournament(self.tournament_size, task)
                parent2 = population.tournament(self.tournament_size, task)
                new_individual = crossover(self.crossover_rate, parent1, parent2)
                new_individual.mutate(self.mutation_rate)
                new_population.add_individual(new_individual)
            best_individual, best_evaluation = population.best(task)
            if best_evaluation > fittest_evaluation:
                fittest_evaluation = best_evaluation
                fittest_individual = best_evaluation
            history.append(best_evaluation)
            population = new_population
        return history, fittest_individual

    def test_param(self, test_cases, param_name, test_set, iterations, task):
        print("\n" + param_name)
        t = tqdm(
            enumerate(test_set),
            desc="Test: 0/" + str(len(test_set)),
            total=len(test_set),
            leave=True,
        )
        param_history = []
        for i, param in t:
            test_histories = []
            if param_name == "Crossover Rate":
                self.crossover_rate = param
            elif param_name == "Mutation Rate":
                self.mutation_rate = param
            elif param_name == "Population Size":
                self.populations_size = param
            elif param_name == "Tournament Size":
                self.tournament_size = param
            else:
                raise Exception("param_name", "wrong")

            for j in range(test_cases):
                history, _ = self.fit(iterations, task)
                test_histories.append(history)
                t.set_description(
                    "Test: {}/{}".format(
                        test_cases * i + j + 1, test_cases * len(test_set)
                    )
                )
                t.refresh()
            param_history.append(test_histories)
        return np.array(param_history)


def plot_history(param_history, param_name, test_set):
    fig = plt.gcf()
    fig.set_size_inches(20, 5)
    fig.suptitle("Evaluation " + param_name, fontsize=20)
    subplots_no = np.array(param_history).shape[0]

    for i, test_histories in enumerate(param_history):
        test_histories_array = np.array(test_histories)
        y = test_histories_array.mean(axis=0)
        x = np.array(range(test_histories_array.shape[1]))
        y_err = test_histories_array.std(axis=0)
        ax = plt.subplot(1, subplots_no, i + 1)
        ax.set_title("{}: {}".format(param_name, test_set[i]), fontsize=15)
        ax.errorbar(
            x=x,
            y=y,
            yerr=y_err,
            color="black",
            elinewidth=15,
            ecolor="red",
            linewidth=5,
            capsize=3,
        )
        ax.set_xlabel("Epochs", fontsize=12)
        ax.set_ylabel("Mean evaluation", fontsize=12)
    plt.show()


def read_task(file_name):
    array = []
    file_name = open(file_name, "r")
    for line in file_name.readlines():
        array.append([float(i) for i in line.split(',')])
    task_array = np.asarray(array)
    return task_array


def generate_task(output_file_path):
    n = np.random.randint(1000, 2000)
    W = np.random.randint(10000, 20000)
    S = np.random.randint(10000, 20000)

    w = np.random.random(n) * 10 * W / n
    s = np.random.random(n) * 10 * S / n
    c = np.random.random(n) * n
    if w.sum() > 2 * W and s.sum() > 2 * S:
        with open(output_file_path, 'w', newline='') as out:
            writer = csv.writer(out)
            writer.writerow([n, W, S])
            for i in range(n):
                writer.writerow([w[i], s[i], c[i]])
    else:
        generate_task(output_file_path)


if __name__ == "__main__":
    generate_task("test.csv")
    generated_task_array = read_task("test.csv")
    task = Task(generated_task_array)
    testing = [("Crossover Rate", [0.1, 0.5, 0.9]),
               ("Mutation Rate", [0.00001, 0.01, 0.1]),
               ("Tournament Size", [5, 20, 70]),
               ("Population Size", [5, 10, 50])
               ]
    for param_name, test_set in testing:
        genetic_algorithm = GeneticAlgorithm(populations_size=75, tournament_size=55, crossover_rate=0.95,
                                             mutation_rate=0.002)
        param_history = genetic_algorithm.test_param(test_cases=3, param_name=param_name,
                                                     test_set=test_set, iterations=100, task=task
                                                     )
        plot_history(param_history, param_name, test_set)
