# genetic_evolution_template
Implemented evolutionary/genetic algorithm for usage with your own custom data. Only suitable if your problem can be evaluated and your goal is to maximize or minimize this value. This genetic algorithm is very useful if you have a lot of attributes with a lot of possible values and a perfect solution is not that easy to find.

## Requirements
1. Your problem (now named individual) have some attributes which values are only numeric (int or float), NO text!
2. If you combine two individuals randomly (for each attribute choose randomly the value from one of the two individuals) you get a legal individual.
3. If you change one value randomly, the new individual is legal, too.

## Usage
You must assign values to the following variables:
* `keys` (list): All available attribute names
* `value_ranges` (list of pairs): For each attribute save the minimum and maximum value. If the values are only integers, save the min and max value as an int, otherwise save them as a float. If the value is not continuous, you have only discrete values, just save all possible values in a tuple.
* `distributions` (list): For each attribute save the distribution. If you have discrete values choose `discrete_choice`, otherwise choose `uniform` for uniformly distributed choice and `log_uniform` for choosing some value v uniformly distributed and return 10**v
* `fitness_func` (function): Function which takes an individual (as a dict with the (key, value)-pairs) and returns the score of this individual. The fitness function should be so good, that a higher score means a better individual and a better solution of the problem (if you have a minimization problem just return the negative value).

After specifying each attribute, you can start your evolution (with default parameter values):
```py
import genetic_evolution_template as evol
evolution_experiment = evol.EvolutionTemplate(keys, value_ranges, distributions, fitness_func)
best_indiv, best_generation, nr_gen = evolution_experiment.evolution()
```

If you want more detailed information, which parameters are available, how to use each parameter and more detail how everything works, just enter the following:

```py
import genetic_evolution_template as evol
help(evol)
```

In this help you will get information about each method of the `EvolutionTemplate` class:
* `Intern params`: This describes only the params of the `EvolutionTemplate` class, which are used in this method and which are very IMPORTANT for this method (nothing which is needed everywhere as the `keys` or the actual `generation`)
* `Args`: All arguments you can pass to this method
* `Returns`: Everything what is returned. If you can see only one value, this value is returned. If multiple values are listed, all these values will be returned in a tuple

## Example
At the bottom of `genetic_evolution_template.py`, you can find a medium sized example for finding the best car (in quotes """). If you enter the marked code, you will get all variables you need to start the evolution. Then try entering the above code and you will get the best car, according to the given fitness function.
