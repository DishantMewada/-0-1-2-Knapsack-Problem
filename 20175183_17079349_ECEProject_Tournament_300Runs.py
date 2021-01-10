
import random
import numpy
import matplotlib.pyplot as plt
from deap import base
from deap import creator
from deap import tools
from deap import algorithms

# problem constants:
# length of bit string to be optimized.
INDIVIDUAL_LENGTH = 100 

# Genetic Algorithm constants:

# Population Size.
POPULATION_SIZE = 500

# Probability for crossover.
P_CROSSOVER = 0.9  

# Probability for mutating an individual.
P_MUTATION = 0.1   

# Number of generations.
MAX_GENERATIONS = 100

# Capturing the best of Individuals.
HALL_OF_FAME_SIZE = 10

# Number of consistent runs.
N_RUNS = 300

# Item Size.
NBR_ITEMS = 100

# Max weight of item.
MAX_WEIGHT = 1000

# Set the random seed:
RANDOM_SEED = 42
random.seed(RANDOM_SEED)

# Create the item dictionary: item name is an integer, and value is a (value, weight) 2-uple.
items = {}

# Create random items and store them in the items' dictionary.
# print('items:')
for i in range(NBR_ITEMS):
    items[i] = (random.randint(1, 10), random.randint(1, 100))
#     print(i,':',items[i],end= ' ')
# print()


toolbox = base.Toolbox()

# create an operator that randomly returns 0,1, or 2:
toolbox.register("zeroOrOneOrTwo", random.randint, 1 , 2)

# define a single objective, maximizing fitness strategy:
creator.create("FitnessMax", base.Fitness, weights=(1.0,))

# create the Individual class based on list:
creator.create("Individual", list, fitness=creator.FitnessMax)
#creator.create("Individual", array.array, typecode='b', fitness=creator.FitnessMax)

# create the individual operator to fill up an Individual instance:
toolbox.register("individualCreator", tools.initRepeat, creator.Individual, toolbox.zeroOrOneOrTwo, NBR_ITEMS)

# create the population operator to generate a list of individuals:
toolbox.register("populationCreator", tools.initRepeat, list, toolbox.individualCreator)


# fitness calculation:
# compute the sum of elements.
def MaxFitness(individual):
    return sum(individual),  # return a tuple

def knapsack(individual):
    weight = 0.0
    value = 0.0
    for i in range(NBR_ITEMS):
        value += items[i][0]*individual[i]
        weight += items[i][1]*individual[i]
    if len(individual) > NBR_ITEMS or weight > MAX_WEIGHT:
        return (value-(weight-MAX_WEIGHT)*2),weight
    else:             
        return (value), weight

toolbox.register("evaluate", knapsack)

# genetic operators:

# Tournament selection with tournament size of 4:
toolbox.register("select", tools.selTournament, tournsize=4)

# Single-point crossover:
toolbox.register("mate", tools.cxOnePoint)

# Flip-bit mutation:
# indpb: Independent probability for each attribute to be flipped
toolbox.register("mutate", tools.mutFlipBit, indpb=1.0/INDIVIDUAL_LENGTH)

# Genetic Algorithm flow:
def main():
    maxList = []
    avgList = []
    minList = []
    stdList = []

    # max value of all runs
    maxValue = []
    # corresponding weight
    maxWeight = []

    for r in range(0, N_RUNS):

        print('\nAt the run:', r)
        # create initial population (generation 0):
        population = toolbox.populationCreator(n=POPULATION_SIZE)

        # define the hall-of-fame object:
        hof = tools.HallOfFame(HALL_OF_FAME_SIZE)

        # prepare the statistics object:
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", numpy.mean)
        stats.register("std", numpy.std)
        stats.register("min", numpy.min)
        stats.register("max", numpy.max)

        # perform the Genetic Algorithm flow:
        population, logbook = algorithms.eaSimple(population, toolbox, cxpb=P_CROSSOVER, mutpb=P_MUTATION,
                                                   ngen=MAX_GENERATIONS,
                                                  stats=stats, halloffame=hof, verbose=True)

        # print Hall of Fame info:
        print("Hall of Fame Individuals = ", *hof.items, sep="\n")
        print("\nBest Ever Individual = ", hof.items[0], "\nFitness: ", r ," ",knapsack(hof.items[0]))

        # hof.items[0] --> (value,weight) 

        # append max value to a list
        maxValue.append(knapsack(hof.items[0])[0])
        # append max weight to a list
        maxWeight.append(knapsack(hof.items[0])[1])

        # Genetic Algorithm is done with this run - extract statistics:
        meanFitnessValues, stdFitnessValues, minFitnessValues, maxFitnessValues  = logbook.select("avg", "std", "min", "max")
        
        # Save statistics for this run:
        avgList.append(meanFitnessValues)
        stdList.append(stdFitnessValues)
        minList.append(minFitnessValues)
        maxList.append(maxFitnessValues)

    print()
    print('Max Value from all the runs:', max(maxValue))

    # len(maxValue) == len(maxWeight)
    # print corresponding weight for max value
    for i in range(len(maxValue)):
        if maxValue[i] == max(maxValue):
            print('Corresponding Weight:',maxWeight[i])

    # Genetic Algorithm is done (all runs) - plot statistics:
    x = numpy.arange(0, MAX_GENERATIONS+1)
    avgArray = numpy.array(avgList)
    stdArray = numpy.array(stdList)
    minArray = numpy.array(minList)
    maxArray = numpy.array(maxList)
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.title('Max and Average Fitness for Knapsack with Tournament Selection - 300 Runs')
    plt.errorbar(x, avgArray.mean(0), yerr=stdArray.mean(0),label="Average",color="Red")
    plt.errorbar(x, maxArray.mean(0), yerr=maxArray.std(0),label="Best", color="Green")
    plt.show()
    
    
if __name__ == '__main__':
    main()