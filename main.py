# coding=utf-8

import numpy as np
import random
import pprint
from initialization import initializeNetwork
from run_dict_model import run_network
from validate import checkNetwork
from copy import deepcopy
import matplotlib.pyplot as plt
import pdb

pp = pprint.PrettyPrinter(indent=2)

class GeneticAlgorithm:
    def __init__(self, population_size, 
                 mr_within_layer, 
                 mr_dropout,
                 mr_layer,
                 mr_layer_input,
                 mr_layer_output,
                 mr_block,
                 mr_subsampling,
                 mr_final,
                 crossover_rate,
                 single_point=True,
                 nr_elites=1):
        
        self.population_size = population_size
        # Mutation rates
        self.mr_within_layer = mr_within_layer  # Swapping within-layer layers
        self.mr_dropout = mr_dropout            # Adding dropout within-layer
        self.mr_layer = mr_layer                # Add random layer
        self.mr_layer_input = mr_layer_input    # Mutating layer inputs
        self.mr_layer_output = mr_layer_output  # Mutating layer outputs
        self.mr_block = mr_block                # Swapping blocks
        self.mr_subsampling = mr_subsampling    # Flipping subsample bit
        self.mr_final = mr_final                # Flipping final layer bit
        
        self.crossover_rate = crossover_rate
        self.single_point = single_point
        self.nr_elites = nr_elites
        
        self.population = self.create_random_population()
        results = [self.fitness(net) for net in self.population]
        self.fitnesses = [result[0] for result in results]
        self.accuracies = [result[1] for result in results]
        
    def get_next_generation(self):
        """
        Return the next generation. Fitnesses of the new generation is stored
        in self.fitnesses. If elitism is used, the top fittest networks
        are always included in the next generation.
        """
        fitnesses = self.fitnesses
        
        elite_idxs = np.array(fitnesses).argsort()[:self.nr_elites]
        nr_random_parents = self.population_size - self.nr_elites
        
        new_population = [self.population[i] for i in elite_idxs]
        
        pop = deepcopy(self.population)
        for i in range(nr_random_parents):
            # Reproduce
            parent_1 = self.selection()
            parent_2 = self.selection()
            child, _ = self.crossover(parent_1,parent_2, self.single_point)
            
            unique = False
            while not unique:
                mutated = self.mutate(child)
                assert checkNetwork(mutated)
                self.check_ids(mutated)
                if mutated not in new_population:
                    new_population.append(mutated)
                    unique = True
            
#==============================================================================
#             # Mutate
#             mutated = self.mutate(child)
#             assert checkNetwork(mutated)
#             self.check_ids(mutated)
#             #pp.pprint(mutated)
#             new_population.append(mutated)
#==============================================================================
        
        assert pop == self.population
        
        results = [self.fitness(net) for net in new_population]
        self.fitnesses = [result[0] for result in results]
        self.accuracies = [result[1] for result in results]
        return new_population
        
    def create_random_population(self):
        """Initialize a random population"""
        pop = [initializeNetwork() for i in range(self.population_size)]
        for network in pop:
            self.check_ids(network)
        return pop
    
    def fitness(self, network, verbose=1):
        """
        Run a network and return the fitness. The fitness is 1/val_loss
            
            network :: dict
                The network representation.
        """
        #return 0.5
        #pp.pprint(network)
        try:
            val_loss, val_acc = run_network(network, compile_mode=False, verbose=verbose)
        except KeyError as e:
            pp.pprint(network)
            raise e
        return val_loss, val_acc
        
    def selection(self, tournament_size=2, p=None):
        """
        Return a random new parent using tournament selection.
        
            tournament_size :: int, optional
                Size of the random tournament. 
                Default = 2 for a binary tournament.
            p :: 1-D array-like, optional
                Probability distributions for selecting candidates.
                Default = None for a uniform distribution.
        """
        parents = np.random.choice(deepcopy(self.population), size=tournament_size, replace=False, p=p)
        fitnesses = []
        for par in parents:
            val_loss, _ = self.fitness(par, verbose=0)
            fitnesses.append(val_loss)
        return parents[np.argmin(fitnesses)]
    
#==============================================================================
#     def selection(self, tournament_size=2, p=None):
#         """
#         Return a random new parent using tournament selection.
#         
#             tournament_size :: int, optional
#                 Size of the random tournament. 
#                 Default = 2 for a binary tournament.
#             p :: 1-D array-like, optional
#                 Probability distributions for selecting candidates.
#                 Default = None for a uniform distribution.
#         """
#         pop = deepcopy(self.population)
#         parent_idxs = np.random.choice(range(len(pop)),size=tournament_size, replace=False)
#         parents = [pop[i] for i in parent_idxs]
#         fitnesses = [self.fitnesses[i] for i in parent_idxs]
#         return parents[np.argmin(fitnesses)]
#==============================================================================
    
    def crossover(self, parent_1, parent_2, single_point=True):
        """
        Create two children by performing crossover with the given parents.
        
            parent_1 :: dict
                Network representation of parent 1
            parent_2 :: dict
                Network representation of parent 2
            single_point :: bool, optional
                Whether to use single-point crossover or generalized crossover.
                For generalized crossover, random parts of each parents are
                chosen for each child. Default = True for single-point crossover.
        """
                
        child_1 = {}
        child_2 = {}
        
        assert len(parent_1)==len(parent_2), "Parents need to be of equal length"
        
        if random.random() < self.crossover_rate:         
            N = len(parent_1)
            
            keys = list(parent_1.keys())
            
            if (single_point):
                # Determine crossover point
                r = np.random.randint(2,N-2)
                # Do crossover
                for i in range(r):
                    child_1[keys[i]] = parent_1[keys[i]]
                    child_2[keys[i]] = parent_2[keys[i]]
                for i in range(r,N):
                    child_1[keys[i]] = parent_2[keys[i]]
                    child_2[keys[i]] = parent_1[keys[i]]
            else:
                # Generate a random boolean list
                mask = np.random.choice([True,False], N)
                for i, b in enumerate(mask):
                    child_1[keys[i]] = parent_1[keys[i]] if b else parent_2[keys[i]]
                    child_2[keys[i]] = parent_2[keys[i]] if b else parent_1[keys[i]]
            
            assert set(child_1.keys()) == set(keys), "Something went wrong generating a child."
            assert set(child_2.keys()) == set(keys), "Something went wrong generating a child."
            
            return child_1, child_2
        else:
            return parent_1.copy(), parent_2.copy()
    
    def random_inputs_for_id(self, block_id):
        """
        Generate a random list of inputs for a given block id.
        Utility function is used in mutation step.
            
            block_id :: int
                The id of the block.
        """
        # For the first block, the input is always [-1]
        if block_id == 0:
            return [-1]
        # For blocks thereafter, inputs can be a random set of earlier inputs
        # or -1
        if random.random() < 0.7:
            inputs_one_hot = np.random.randint(2,size=block_id)
            inputs = [i for i in range(block_id) if inputs_one_hot[i] > 0]
        else:
            inputs = [-1]
        
        # If none of the ids was selected, just connect the layer before
        if len(inputs) == 0:
            inputs = [block_id-1]
            
        assert not (-1 in inputs and len(inputs) > 1), "Input -1 can't occur with other inputs."
        return inputs
    
    def check_ids(self,network):
        for block_name in network.keys():
            if "block" in block_name:
                conv_blocks = network[block_name]['convblocks']
                ids = [conv_block['id'] for conv_block in conv_blocks]
                assert list(set(ids)) == sorted(ids),"Ids are incorrect"
    
    def mutate(self, network):
        """
        Mutate a network given the probabilities defined in the initialization
        function.
        
        The possible mutations are:
            - flipping the subsampling/final boolean,
            - removing or adding convolution blocks,
            - adding or removing dropout,
            - changing the order of convolution blocks,
            - changing the inputs of convolution blocks,
            - changing the outputs of the big block,
            - changing the order of big blocks.
        """
        conv_block_keys = []
        for block_name in network.keys():
            if "subsample" in block_name:
                # Mutate subsampling method
                if random.random() < self.mr_subsampling:
                    network[block_name] = not network[block_name]
            elif "final" in block_name:
                if random.random() < self.mr_final:
                    network[block_name] = not network[block_name]
            else:
                conv_block_keys.append(block_name)
                # First, remove or add random conv blocks
                conv_blocks = network[block_name]['convblocks']
                init_inputs_from_id = -1
                if len(conv_blocks) > 1:
                    if random.random() < self.mr_layer:
                        # We can remove a conv block
                        to_remove = np.random.randint(0,len(conv_blocks))
                        #pp.pprint(conv_blocks)
                        #print("Removed convblock {} from {}".format(to_remove,block_name))
                        conv_blocks = [conv_blocks[i] for i in range(len(conv_blocks)) if conv_blocks[i]['id'] != to_remove]
                        
                        # Also remove it from the outputs of the network
                        outputs = [out for out in network[block_name]['output'] if out != to_remove]
                        
                        network[block_name]['output'] = outputs
                        
                        # Decrease ids for layers after removed layer
                        for i in range(len(conv_blocks)):
                            if conv_blocks[i]['id'] > to_remove:
                                # Decrease id in the output
                                try:
                                    out_idx = outputs.index(conv_blocks[i]['id'])
                                    network[block_name]['output'][out_idx] -= 1
                                except:
                                    pass
                                # Decrease id in the layer
                                conv_blocks[i]['id'] -= 1
                                if conv_blocks[i]['id'] == 0:
                                    conv_blocks[i]['input'] = [-1]
                                
                        init_inputs_from_id = to_remove
                        
                if len(conv_blocks) < 3:
                    if random.random() < self.mr_layer:
                        # We can add a conv block
                        max_id = max([conv_block['id'] for conv_block in conv_blocks])
                        if random.random() < 0.5:
                            layers = list(np.random.permutation(['r','d','b','c']))
                        else:
                            layers = list(np.random.permutation(['r','b','c']))
                        self_id = np.random.randint(0,max_id+2)
                        
                        #pp.pprint(conv_blocks)
                        #print("Added convblock {} to {}".format(self_id, block_name))
                        
                        # Increase ids for layers after inserted layer
                        for i in range(len(conv_blocks)):
                            if conv_blocks[i]['id'] >= self_id:
                                conv_blocks[i]['id'] += 1
                        block = {'id':self_id,'layers':layers,'input':[-1]}
                        conv_blocks.append(block)
                        #pp.pprint(conv_blocks)
                        if init_inputs_from_id > -1:
                            init_inputs_from_id = min(self_id,init_inputs_from_id)
                        else:
                            init_inputs_from_id = self_id
                
                
                # If some layers were added or removed, we always randomly 
                # initialize inputs of later layers
                if init_inputs_from_id > -1:
                    for i in range(len(conv_blocks)):
                        block_id = conv_blocks[i]['id']
                        if block_id >= init_inputs_from_id:
                            conv_blocks[i]['input'] = self.random_inputs_for_id(block_id) 
                # End of code removing/adding conv blocks
                
                #ids = [conv_block['id'] for conv_block in conv_blocks]
                #assert list(set(ids)) == sorted(ids),"Ids are incorrect"
                
                network[block_name]['convblocks'] = []
                           
                # Within-conv_block mutations
                for i, conv_block in enumerate(conv_blocks):
                    # Add or remove dropout
                    if random.random() < self.mr_dropout:
                        # Check if there is already dropout
                        if 'd' in conv_block['layers']:
                            # Remove it
                            conv_block['layers'] = [layer for layer in conv_block['layers'] if layer != 'd']
                            assert 'd' not in conv_block['layers']
                        else:
                            # Add it
                            conv_block['layers'].append('d')
                            assert 'd' in conv_block['layers']
                    # Then, mutate the order of layers
                    if random.random() < self.mr_within_layer:
                        conv_block['layers'] = list(np.random.permutation(conv_block['layers']))
                    # Then, mutate the inputs of layers
                    if random.random() < self.mr_layer_input:
                        # Layers can only take inputs from layers with lower id
                        conv_block['input'] = self.random_inputs_for_id(conv_block['id'])
                    network[block_name]['convblocks'].append(conv_block)
                
                #self.check_ids(network)
                
                
                # Mutate which layers are channeled to output
                max_id = max([conv_block['id'] for conv_block in conv_blocks])
                outputs = network[block_name]['output']
                for i in range(max_id):
                    if random.random() < self.mr_layer_output:
                        # If this output is in the outputs, remove it
                        # Otherwise, add it
                        if i in outputs:
                            outputs.remove(i)
                        else:
                            outputs.append(i)
                    if len(outputs) == 0:
                        outputs= [max_id]
                network[block_name]['output'] = outputs
                
                #self.check_ids(network)
                # Check which layers are dead ends
                connected_ids = []
                for conv_block in conv_blocks:
                    connected_ids.extend(conv_block['input'])
                connected_ids.extend(network[block_name]['output'])
                connected_ids = set(connected_ids)
                network[block_name]['output'].extend(
                        [conv_block['id'] for conv_block in conv_blocks if conv_block['id'] not in connected_ids]
                        )
                #self.check_ids(network)
                
                
        # Permute blocks
        #self.check_ids(network)
        if random.random() < self.mr_block:
            perm = list(np.random.permutation(list(conv_block_keys)))
            new_network = network.copy()
            keys = list(conv_block_keys)
            for key, old_key in zip(perm,keys):
                new_network[key] = network[old_key]
            network = new_network
        #self.check_ids(network)
        return network
    
if __name__ == '__main__':
    print("Start Initial Generation")
    ga = GeneticAlgorithm(population_size=20, 
                 mr_within_layer=0.3, 
                 mr_dropout=0.3,
                 mr_layer=0.3,
                 mr_layer_input=0.3,
                 mr_layer_output=0.3,
                 mr_block=0.3,
                 mr_subsampling=0.3,
                 mr_final=0.3,
                 crossover_rate=0.3,
                 single_point=True,
                 nr_elites=2)
    
    num_iterations = 20
    best_fitnesses = []
    avg_fitnesses = []
    best_accs = []
    avg_accs = []
    
    # Initial fitness & accs
    best_fitnesses.append(min(ga.fitnesses))
    avg_fitnesses.append(np.mean(ga.fitnesses))
    best_accs.append(max(ga.accuracies))
    avg_accs.append(np.mean(ga.accuracies))
    print("Finished Initial Generation")
    print("best_loss: {0:.3f}".format(min(ga.fitnesses)))
    print("avg_loss: {0:.3f}".format(np.mean(ga.fitnesses)))
    print("best_acc: {0:.3f}".format(max(ga.accuracies)))
    print("avg_acc: {0:.3f}".format(np.mean(ga.accuracies)))
    
    # Start GA
    for i in range(num_iterations):
        print('')
        print("Start Generation {}".format(i+1))
        ga.population = ga.get_next_generation()
        best_fitnesses.append(min(ga.fitnesses))
        avg_fitnesses.append(np.mean(ga.fitnesses))
        best_accs.append(max(ga.accuracies))
        avg_accs.append(np.mean(ga.accuracies))
        print("Finished Generation {}:".format(i+1))
        print("best_loss: {0:.3f}".format(min(ga.fitnesses)))
        print("avg_loss: {0:.3f}".format(np.mean(ga.fitnesses)))
        print("best_acc: {0:.3f}".format(max(ga.accuracies)))
        print("avg_acc: {0:.3f}".format(np.mean(ga.accuracies)))
        
    plt.plot(best_fitnesses,label='Best Fitness')
    plt.plot(avg_fitnesses,label='Mean Fitness')
    plt.plot(best_accs,label='Best Accuracy')
    plt.plot(avg_accs,label='Mean Accuracy')
    plt.legend()
    plt.show()