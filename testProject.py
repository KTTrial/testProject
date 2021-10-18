from dwave.system import EmbeddingComposite, DWaveSampler
import numpy as np
import dimod
import re
import dwave.inspector

def get_matrix(path):
    grid=[]
    print(path)
    with open(path,"r") as f:
        for line in f:
            w = list(map(str,line[:-1].split("\t")))
            grid.append(w)
    return np.array(grid)
def most_frequent(x):
    return max(set(x), key = x.count)

def build_bqm(cost):
    #bqm = dimod.BQM('BINARY')
    #bqm = dimod.BQM("BINARY")
    #dimod.binary.
    lagrange = 8

    # Add the linear biases
    linear_terms = {}
    q_terms = {}
    for i in range(len(cost)):
        for j in range(len(cost[0])):
            # The linear terms have a contribution from the objective (cost) and constraint (-lagrange)
            #bqm.add_linear(f'x_{i}_{j}', cost[i][j] - lagrange)
            termj = "fx"+str(i)+str(j)
            linear_terms[termj] = cost[i][j] - lagrange

            # The quadratic terms come from the constraint
            for k in range(j+1, len(cost[0])):
                #bqm.add_quadratic(f'x_{i}_{j}', f'x_{i}_{k}', 2*lagrange)
                termk = "fx"+str(i)+str(k)
                q_terms[(termj, termk)] =  2*lagrange
    print("linear terms:" + str(linear_terms))
    print("q terms:" + str(q_terms))
    bqm = dimod.BQM(linear_terms, q_terms, 1.4, dimod.Vartype.BINARY)    
    return bqm



if __name__ == '__main__':
    preference = get_matrix("preference.txt")
    match = get_matrix("match.txt")

    rows = preference.shape[0]
    cols = preference.shape[1]
    cost = np.array([[None] * cols] * rows)

    # Build the cost matrix
    for i in range(rows):
        for j in range(cols):
            if match[i][j] == '0':
                cost[i][j] = 99
            else:
                cost[i][j] = int(preference[i][j])

    # Build the BQM
    bqm = build_bqm(cost)

    # Submit to the QPU
    sampler = EmbeddingComposite(DWaveSampler())
    ss = sampler.sample(bqm, num_reads=50)

    samples = ss.samples(5, 'energy')
    print("solution 0:")
    print(samples[0])
    print("solution 1:")
    print(samples[1])


    # Process results
    #print("\nRESULTS")
    #print("\nStudent, Tutor")
    #print(ss.first.sample)
    #for s in ss.first.sample:
    #    if ss.first.sample[s] == 1:
    #         print(ss.first.sample[s])
#            pair = re.findall('[0-9]+', s)
#            print(pair[0], ' '*8, pair[1])



