import random
import sys
from functools import reduce

class EvolutionTemplate:
    def __init__(self, keys, value_ranges, distributions, fitness_func=None, *, \
        nr_individuals=20, max_gen_nr=100, max_same_gen=15, \
        lambd=0.7, alpha=2, beta=5, rel_diff=0.00001, \
        nr_best=5, nr_rndm=3, nr_breed=8, nr_mutate=4):
        """Working with genetic algorithms and applying all kinds of methods.
        Create the first randomly created generation.

        Args:
            Positional:
                keys: list. keys, each individual must have
                value_ranges: list of tuples. For each key, you need a tuple at
                    the same position as in keys, which saves the min and max
                    for this key (int or float). If you have a 'discrete_choice'
                    distribution for a specific key, you can have more than two
                    values in this key-specific tuple
                distributions: list of str. For each key, you need a
                    distribution at the same position as in keys, which saves
                    the random distribution of the values. 'uniform',
                    'log_uniform' and 'discrete_choice' are the only possible
                    arguments.

                    'uniform' and 'log_uniform' takes one random
                    value uniformly between min and max value.
                    'log_uniform' returns 10**rndm_val.
                    'discrete_choice' returns randomly one of the given values.

            Keyword:
                nr_individuals: int. Number of individuals for each generation
                max_gen_nr: int. Maximum number of generations before aborting
                    evolution
                max_same_gen: int. Maximum number of generations you need the
                    same fitness score for the best individual, before stopping
                    evolution early
                lambd: float. The number of changed attributes is exponentially
                    distributed with lambd as lambda value (lambda is reserved
                    keyword)
                alpha, beta: numeric, alpha > 0, beta > 0. Percentage change of
                    an attribute is beta distributed with arguments alpha and
                    beta.
                fitness_func: func. Fitness function, which accepts one
                    individual as a parameter and returns the fitness of this
                    individual
                rel_diff: float, 0 <= rel_diff <= 1. Maximum percentage change
                    of highest fitness_score which is leading to zero
                    difference. Important for early stopping evolution
                nr_best: int. Number of top individuals, which must go into the
                    next generation
                nr_rndm: int. Number of random individuals, which go into the
                    next generation
                nr_breed: int,
                    nr_breed <= (nr_best+nr_rndm)*(nr_best+nr_rndm-1)/2.
                    Number of breeded individuals, with two parents from the
                    individuals generated with the nr_best and nr_rndm
                    arguments.
                nr_mutate: int,
                    nr_mutate == nr_individuals - nr_best - nr_rndm - nr_breed
                    Number of mutated individuals, where the original
                    individuals are chosen from the individuals generated with
                    the nr_best and nr_rndm arguments.

        Returns: Nothing
        """

        # Initialize all values
        self.keys, self.value_ranges, self.distributions = keys, value_ranges, distributions
        self.nr_individuals, self.max_gen_nr, self.max_same_gen = nr_individuals, max_gen_nr, max_same_gen
        self.lambd, self.alpha, self.beta, self.fitness_func = lambd, alpha, beta, fitness_func
        self.nr_best, self.nr_rndm, self.nr_breed, self.nr_mutate = nr_best, nr_rndm, nr_breed, nr_mutate
        self.generation, self.rel_diff = None, rel_diff

        # Look if all values are valid
        if self.nr_best < 0 or self.nr_rndm < 0 or self.nr_breed < 0 or self.nr_mutate < 0:
            raise ValueError("A number for creating new generations is < 0!")
        if self.nr_breed > (self.nr_best+self.nr_rndm)*(self.nr_best+self.nr_rndm+1)/2:
            raise ValueError("Number of breeds is bigger than all possible breeds. " +\
                +"Every pair of parents can create only one child!")
        if self.nr_mutate != self.nr_individuals-self.nr_best-self.nr_rndm-self.nr_breed:
            raise ValueError("Total number of created individuals doesn't fit with the total number of individuals.")

        # Create first random generation
        self.create_random_generation()

    def create_random_generation(self):
        """Creates a random generation.

        Intern params:
            nr_individuals: Number of created individuals for one generation
            keys: All keys each individual must have
            value_ranges: Value ranges for each key
            distributions: Distribution of choosing elements from value_ranges.
                uniform: One value of value_ranges is chosen uniformly
                log_uniform: One value v of value_ranges is chosen uniformly
                    and 10**v is returned
                discrete_choice: One value of the values in value_ranges

        Returns:
            generation: For each individual and for each key, one value in
                value_ranges will be selected randomly regarding the
                distributions. After creating enough individuals, the whole
                randomly created generation will be returned.
        """

        generation = []
        for i in range(self.nr_individuals):
            generation.append(dict())
            for j in range(len(self.keys)):
                key = self.keys[j]

                # For discrete_choice take some random choice
                if self.distributions[j] == 'discrete_choice':
                    generation[i][key] = random.choice(self.value_ranges[i])
                else:
                    min_val, max_val = self.value_ranges[j]

                    # Get the type of values by regarding type of min_val and max_val
                    if type(min_val) != type(max_val):
                        raise TypeError("min_val and max_val does not have the same type")
                    elif type(min_val) == int:
                        generation[i][key] = random.randint(min_val, max_val)
                    else:
                        generation[i][key] = random.uniform(min_val, max_val)

                    # For log_uniform return 10**rndm_val
                    if self.distributions[j] == 'log_uniform':
                        generation[i][key] = 10**generation[i][key]
        self.generation = generation
        return generation

    def breed(self, parents):
        """Breed some individuals to create a new one.

        Args:
            parents: List of individuals which will pass their genes
                (the values of their attributes) to their child

        Returns:
            child: Individual created from his parents. For each key one value
                from a random parent is taken.
        """
        # testing if the keys of all parents are the same
        if len([parent.keys() for parent in parents if parent.keys() != parents[0].keys()]) != 0:
            raise KeyError("Not all parents have the same keys!")
        child = dict()

        # take the value from one random parent for each key
        for key in parents[0]:
            child[key] = random.choice([parent[key] for parent in parents])
        return child

    def mutate(self, individual):
        """Mutates an individual.

        Intern params:
            lambd: get a random number of attributes which should be changed
                with the exponential distribution with param lambd
            alpha, beta: get a random percentage how strong the actual value
                should be changed with the beta distribution and params alpha,
                beta

        Args:
            individual: Individual which should be mutated a little bit.

        Returns:
            individual: Randomly choose how many attributes should be changed.
                Which attributes are changed is chosen randomly. For each
                changing attribute, a percentage p and a sign s (-1 or 1) are
                chosen randomly. The actual value v is modified by adding
                s*p*v to v. If you reach the min_val or max_val, the modified
                value will be scaled down to stay inside this interval.
        """
        num_changes = int(random.expovariate(self.lambd)) + 1
        if num_changes > len(individual):
            num_changes = len(individual)

        # create the keys which should be modified randomly
        changed_keys = random.sample([i for i in range(len(individual))], num_changes)
        for i in changed_keys:
            key = list(individual.keys())[i]
            min_val, max_val = self.value_ranges[i]

            # create a random sign
            random_sign = random.choice([-1, 1])
            if individual[key] == min_val:
                random_sign = 1
            elif individual[key] == max_val:
                random_sign = -1

            # create a random percentage
            random_percentage = random.betavariate(self.alpha, self.beta)
            add_val = random_sign * random_percentage * individual[key]

            # look for the type and parse the random value to int if necessary
            if type(individual[key]) == int:
                if int(add_val) == 0:
                    add_val = random_sign
                else:
                    add_val = int(add_val)
            individual[key] += add_val

            # make sure that all values are inside the allowed interval
            if individual[key] > max_val:
                individual[key] = max_val
            elif individual[key] < min_val:
                individual[key] = min_val
        return individual

    def fitness(self, generation):
        """Calculate fitness for each individual and sort them descending by
        fitness.

        Intern params:
            fitness_func: Fitness function to calculate the fitness of each
                individual (higher fitness score means a better individual)

        Args:
            generation: list of individuals which should be evaluated

        Returns:
            sorted_generation: Input generation but sorted descending by
                fitness values.
            sorted_fitness_scores: All fitness scores sorted descending and
                corresponding to the values of the sorted_generation
        """
        # create scores, if you have a fitness_function, otherwise take 0 for each individual
        if type(self.fitness_func) != type(None):
            fitness_scores = [self.fitness_func(indiv) for indiv in generation]
        else:
            fitness_scores = [0 for indiv in generation]

        # add the indizes to the fitness_score, to not loose the position of the individuals
        fitness_scores_indizes = [(fitness_scores[i], i) for i in range(len(fitness_scores))]

        # sort the generation descending by fitness_score
        sorted_generation = [generation[fit[1]] for fit in sorted(fitness_scores_indizes, reverse=True)]
        return sorted_generation, sorted(fitness_scores, reverse=True)

    def create_next_generation(self, old_generation, sorted=True):
        """Create next generation

        Intern params:
            nr_best: Number of best individuals which go into next generation
            nr_rndm: Number of random individuals which go into next generation
            nr_breed: Number of breeded individuals, each one breeded from two
                parents which are created from the above two params, which go
                into next generation
            nr_mutate: Number of mutated individuals, mutated only individuals
                which are created from the nr_best and nr_rndm params, which
                go into next generation

        Args:
            old_generation: The old generation which should be updated
            sorted: Indicates if the old_generation is already sorted
                descending by fitness score

        Returns:
            new_generation: New generation with the same number of individuals
                as the previous generation, but some changes according to
                the above arguments.
        """

        # Sort the old_generation by fitness score and save the first nr_best examples
        if not sorted:
            new_generation = self.fitness(old_generation)[0][:self.nr_best]
        else:
            new_generation = old_generation[:self.nr_best]

        # add the random individuals
        rndm_individuals = random.sample(old_generation[self.nr_best:], self.nr_rndm)
        new_generation.extend(rndm_individuals)

        # breeding two parents, which are randomly selected (but not breeding twice)
        parent_pairs = [(new_generation[i], new_generation[j]) for i in range(len(new_generation)) for j in range(i+1, len(new_generation))]
        chosen_parent_pairs = random.sample(parent_pairs, self.nr_breed)
        children = [self.breed(parents) for parents in chosen_parent_pairs]
        new_generation.extend(children)

        # mutate only some generated individuals
        chosen_mutations = random.sample(new_generation[self.nr_best+self.nr_rndm:], self.nr_mutate)
        mutations = [self.mutate(mut) for mut in chosen_mutations]
        new_generation.extend(mutations)
        self.generation = new_generation
        return new_generation

    def end_evolution(self, fitness_scores):
        """Indicates if there is almost no improvement and the evolution can be
        stopped.

        Intern params:
            max_same_gen: Maximum number of generations where the same
                highscore is reached
            rel_diff: Maximum relative difference between the highscores of the
                last and the max_same_gen-last generation to say, that there
                was no improvement and the evolution can be stopped

        Args:
            fitness_scores: A list of lists, where each list contains all
                fitness_scores of this generation and the last list are the
                fitness_scores of the newest generation

        Returns:
            bool: True if there is no (or less than rel_diff%) improvement and
                the evolution can be stopped, False otherwise
        """
        # If not enough generations, don't end the evolution
        if len(fitness_scores) < self.max_same_gen:
            return False

        # add the difference of the last max_same_gen scores together
        fitness_scores = fitness_scores[-self.max_same_gen:]
        first_scores = [score[0] for score in fitness_scores]
        total_scores = reduce(lambda x, y: x + abs(first_scores[0] - y), first_scores, 0)

        # end evolution, when the best score doesn't change (or only about rel_diff%) in the last n iterations
        if total_scores == 0 or abs(total_scores) < abs(first_scores[0])*self.rel_diff:
            return True
        return False

    def print_generation(self, generation, fitness_scores, gen_nr):
        """Print a generation pretty in the command line.

        Args:
            generation: The whole generation
            fitness_scores: The fitness scores of the generation corresponding
                to the generation
            gen_nr: Number of generation

        Returns: Nothing
        """

        write_line = 'Generation '+str(gen_nr)+':\n'

        # Make the first line to indicate the name of each column
        for key in self.keys:
            write_line += key+' | '
        write_line += '\n'

        # print each individual, its values and its score
        for i in range(len(generation)):
            indiv = generation[i]
            write_line += str(tuple(indiv.values()))+",    Score: "+str(fitness_scores[i]) + '\n'
        sys.stdout.write(write_line)
        sys.stdout.flush()

    def evolution(self):
        """Make the whole evolution until the best individual is found or there
        is no more improvement in the fitness score.

        Args: Nothing
        Returns:
            sorted_generation: Whole sorted generation at the end of evolution
            best_individual: Individual with best fitness score
            gen_nr: Number of generations done.
        """

        gen_nr = 0
        sorted_generation, fitness_scores = self.fitness(self.generation)
        last_fitness_scores = [fitness_scores]

        # Repeat evolution until the end is reached
        while gen_nr < self.max_gen_nr and not self.end_evolution(last_fitness_scores):
            # create and evaluate the next generation
            self.print_generation(sorted_generation, fitness_scores, gen_nr)
            generation = self.create_next_generation(sorted_generation)
            sorted_generation, fitness_scores = self.fitness(generation)

            # add the new fitness_scores to the last_fitness_scores list
            last_fitness_scores.append(fitness_scores)

            # to save some space, save only the important fitness_scores
            # this are the last max_same_gen scores, to know if there is enough
            # change to continue evolution or not
            if len(last_fitness_scores) > self.max_same_gen:
                last_fitness_scores.remove(last_fitness_scores[0])
            gen_nr += 1
        self.generation = sorted_generation
        return sorted_generation, sorted_generation[0], gen_nr

"""Example attributes for calculating the optimal car:

=============================ENTER=THIS=PYHTHON=CODE=======================================

keys = ['nr_seats', 'driven_years', 'total_fuel', 'fuel_per_100', 'PS', 'acceleration', 'start_price']
value_ranges = [(2, 5), (0, 10), (20, 150), (2.0, 20.0), (50, 850), (1.8, 18.0), (5000, 20000)]
distributions = ['uniform' for i in range(len(keys))]
fitness_func = calc_fitness

def calc_fitness(indiv):
    # percentage loss of car value after x years:
    # high loss in the first years and stabilized loss of 5% per year after 5 years
    value_loss = [0, 0.3, 0.37, 0.45, 0.51, 0.57]
    value_loss.extend([0.63 + 0.05*i for i in range(5)])

    # calculate fitness
    # behind each calculated value, the average value and optimization direction (min or max this value for getting the best car)
    # is written as a comment. The average value is a simply and logically guess (by trying some average values I can think of)

    # higher total distance is much better
    total_distance = indiv['total_fuel'] / indiv['fuel_per_100'] # 10, max

    # lower loss after 5 years is better
    five_years_value_loss = 0.25 if indiv['driven_years'] > 5 else \
        value_loss[indiv['driven_years']+5] - value_loss[indiv['driven_years']] # 0.25-0.5 min

    # lower price per PS is preferable
    price_power = indiv['start_price'] / indiv['PS'] # 100, min

    # combining the PS with the acceleration and fuel needed for 100km
    # Higher PS and lower acceleration and lower fuel is better
    fuel_power = indiv['PS'] / indiv['acceleration'] / indiv['fuel_per_100'] # 1-2, max

    # Higher cost is better because it's probably better quality
    good_quality = indiv['start_price'] # 100000, max

    # Higher seat numbers are preferable
    # nr_seats 2-5, max

    # generate reward by standardize all values (the original average value is chosen logically
    # without any reference points) to get a somehow equal evaluation of all attributes
    reward = total_distance / 10 - five_years_value_loss * 3 - price_power / 100 + \
        fuel_power * 0.7 + indiv['nr_seats'] / 10 - indiv['driven_years'] / 5 + good_quality / 100000
    return reward

===================================================================================================
"""
