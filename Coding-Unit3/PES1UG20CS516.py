# SRN:PES1UG20CS516
# VRUSHANK G
# SEC-I

import numpy as np
import pygad
import pygad.nn
import pygad.gann

def fit_value(sol,idx_solution):
    global GANN_instance, input, output

    value_pred = pygad.nn.predict(last_layer=GANN_instance.population_networks[idx_solution],data_inputs=input)
    corr_pred = np.where(value_pred == output)[0].size
    final_value = (corr_pred/output.size)*100
    return final_value

def callback_generation(GA):
    global GANN_instance
    pop_mat = pygad.gann.population_as_matrices(population_networks=GANN_instance.population_networks,population_vectors=GA.population)
    GANN_instance.update_population_trained_weights(population_trained_weights=pop_mat)
    print("GA Number = ",GA.generations_completed,end=" ")
    print("Accuracy% = ",GA.best_solution()[1])

input = np.array([[1, 1],[1, 0],[0, 1],[0, 0]])
output = np.array([0,1,1,0])

GANN_instance = pygad.gann.GANN(num_solutions=10,num_neurons_input=2,num_neurons_hidden_layers=[2],
                                num_neurons_output=2,hidden_activations=["relu"],output_activation="softmax")

population_vectors = pygad.gann.population_as_vectors(population_networks=GANN_instance.population_networks)

ga_instance = pygad.GA(num_generations=50,num_parents_mating=3, initial_population=population_vectors.copy(),
                       fitness_func=fit_value,mutation_percent_genes=10,callback_generation=callback_generation)
ga_instance.run()
sol,val,idx_sol = ga_instance.best_solution()
print("Solution :{solution} ".format(solution=sol))
print("Value :{Final_Value}".format(Final_Value=val))
print("IDX Solution :{Sol_IDX}".format(Sol_IDX=idx_sol))