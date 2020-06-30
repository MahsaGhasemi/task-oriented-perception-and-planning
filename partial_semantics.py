import sys
import time
from math import isclose
import numpy as np
import copy
import itertools
from scipy.special import comb
from scipy.stats import entropy
from scipy.spatial import distance
import random
import gurobipy as grb
import pandas as pd
import matplotlib.pyplot as plt

class MC:
    """Generate an MDP object for a problem"""

    def __init__(self, origin=None, mdp=None, policy=None, *args, **kwargs):

        self.origin = origin

        if self.origin == 'mdp':
            self.induced_mc(mdp, policy)
        elif self.origin == 'book':
            self.book_ex()
        elif self.origin == None:
            pass
        else:
            raise NameError("Given Markov chain origin is not supported")

    def book_ex(self):
        """Create a test MC from model-checking book Figure 10.1"""

        self.states = np.arange(4) # set of states {0 : 'start', 1 : 'try', 2 : 'lost', 3 : 'delivered'}
        self.init_state = 0 # initial state
        self.transitions = np.array([
                [0,1,0,0],
                [0,0,0.1,0.9],
                [0,1,0,0],
                [1,0,0,0]], dtype=np.float64) # transition function

    def induced_mc(self, mdp, policy):
        """Generate the inuced MC from an MDP and a policy"""

        self.states = mdp.states # set of states
        self.init_state = mdp.init_state # initial state
        self.transitions = np.zeros((len(self.states),len(self.states)), dtype=np.float64) # transition function
        if policy.memory_use:
            raise NameError("Induced Markov chain for policies with memory is not supported")
        else:
            if policy.randomization:
                for state in self.states:
                    assert np.sum(policy.mapping[state]) == 1, "Policy has improper distribution in state %i" % (state)

                    for ind_a, action in enumerate(mdp.actions[mdp.enabled_actions[state]]):
                        self.transitions[state] += [policy.mapping[state][ind_a] *
                                mdp.transitions[state,action,next_state]
                                for next_state in self.states]
            else:
                for state in self.states:
                    action = policy.mapping[state]
                    assert mdp.enabled_actions[state,action], \
                           "Policy takes invalid action %i in state %i" % (action,state)
                    self.transitions[state] = [mdp.transitions[state,action,next_state]
                                               for next_state in self.states]

    def make_absorbing(self, state):
        """Make a state of the MC absorbing"""

        self.transitions[state,:] = np.zeros(len(self.states)) # remove all transitions
        self.transitions[state,state] = 1 # add self transition


class Policy:
    """Generate a policy object for an MDP"""

    def __init__(self, mdp, randomization=False, memory_use=False, mapping=None):

        self.mdp = mdp
        self.randomization = randomization
        self.memory_use = memory_use
        if mapping == None:
            self.mapping = dict()
        else:
            self.mapping = mapping

    def unif_rand_policy(self):
        """Generate a random policy with uniform distribution"""

        self.randomization = True
        self.mapping = {s : np.full(np.sum(self.mdp.enabled_actions[s]),
                                    1.0/np.sum(self.mdp.enabled_actions[s]))
                        for s in self.mdp.states}
        # note that mapping is only defined for the enabled actions

    def rand_policy(self):
        """Generate a random policy"""

        if self.memory_use:
            raise NameError("Random policy with memory is not defined")
        else:
            if self.randomization:
                self.mapping = dict()
                for s in self.mdp.states:
                    self.mapping[s] = np.diff(np.concatenate(
                            ([0], np.sort(np.random.uniform(0,1,
                             np.sum(self.mdp.enabled_actions[s])-1)),[1])))
            else:
                self.mapping = {s : np.random.randint(0,np.sum(self.mdp.enabled_actions[s]))
                                for s in self.mdp.states}

    def take_action(self, state, memory=None):
        """Select a single action according to the policy"""

        if self.memory_use:
            pass
        else:
            if self.randomization:
                action = np.random.choice(self.mdp.actions[self.mdp.enabled_actions],
                                          p=self.mapping[state])
                next_state = np.random.choice(self.mdp.states,
                                              p=self.mdp.transitions[state,action])
            else:
                action = self.mapping[state]
                next_state = np.random.choice(self.mdp.states,
                                              p=self.mdp.transitions[state,action])

        return (action, next_state)

    def simulate(self, init_state, n_step):
        """Simulate trajectory realizations of a policy"""

        trajectory = np.empty(n_step+1, dtype=np.int32)
        trajectory[0] = init_state
        action_hist = np.empty(n_step, dtype=np.int32)

        for step in range(n_step):
            (action_hist[step], trajectory[step+1]) = self.take_action(trajectory[step])

        return (trajectory, action_hist)

    def verify_trajectory(self, trajectory, spec):
        """Check whether a single trajectory satisfies a specification"""

        if len(spec) == 2:
            # reach-avoid specification
            assert len(set.intersection(set(spec[0]),set(spec[1]))) == 0, \
                   "The specification creates conflict"
            reach_steps = []
            for r in spec[1]:
                if len(np.nonzero(trajectory==r)[0]) > 0:
                    reach_steps.append(np.nonzero(trajectory==r)[0][0])
            reach_min = min(reach_steps)

            if len(reach_steps) == 0:
                return False
            else:
                for t in range(reach_min):
                    if trajectory[t] in spec[0]:
                        return False
                return True

        else:
            raise NameError("Given specification is not handled")

    def evaluate(self, spec, s_current):
        """Evaluate a policy with respect to a specification"""

        ind_mc = MC(origin='mdp', mdp=self.mdp, policy=self)
        (vars_val,_) = verifier(copy.deepcopy(ind_mc), spec)

        return vars_val[s_current]


class MDP:
    """Generate an MDP object for a problem"""

    def __init__(self, problem_type, *args, **kwargs):

        self.problem_type = problem_type

        if self.problem_type == 'simple':
            self.simple_problem()
        elif self.problem_type == 'book':
            self.book_ex()
        elif self.problem_type == 'gridworld':
            self.gridworld_2D()
        else:
            raise NameError("Given problem type is not supported")

    def simple_problem(self):
        """Create a simple MDP with small size"""

        self.states = np.arange(3) # set of states
        self.init_state = 0 # initial state
        self.actions = np.arange(3) # set of actions
        self.action_mapping = {0 : 'a', 1 : 'b', 2 : 'c'}
        self.enabled_actions = np.array([[1,1,0],[1,0,1],[0,1,1]], dtype=np.bool) # enabled actions in states
        self.transitions = np.array([
                [[0,1,0],[0,0,1],[0,0,0]],
                [[1,0,0],[0,0,0],[0,1,0]],
                [[0,0,0],[1,0,0],[0,0,1]]], dtype=np.float64) # transition function

    def book_ex(self):
        """Create a test MDP from model-checking book Figure 10.21"""

        self.states = np.arange(8) # set of states
        self.init_state = 0 # initial state
        self.actions = np.arange(2) # set of actions
        self.action_mapping = {0 : 'alpha', 1 : 'beta'}
        self.enabled_actions = np.array([[1,1],
                                         [1,0],
                                         [1,0],
                                         [1,0],
                                         [1,0],
                                         [1,1],
                                         [1,0],
                                         [1,0]], dtype=np.bool) # enabled actions in states
        self.transitions = np.zeros((len(self.states),len(self.actions),
                                     len(self.states)), dtype=np.float64) # transition function
        self.transitions[0,0,[1]] = [1]; self.transitions[0,1,[2,4]] = [1/3,2/3]
        self.transitions[1,0,[1,2,3]] = [1/2,7/18,1/9]
        self.transitions[2,0,[2]] = [1]
        self.transitions[3,0,[3]] = [1]
        self.transitions[4,0,[5,6]] = [1/4,3/4]
        self.transitions[5,0,[6]] = [1]; self.transitions[5,1,[2,7]] = [1/3,2/3]
        self.transitions[6,0,[5,6]] = [2/5,3/5]
        self.transitions[7,0,[2,3]] = [1/2,1/2]

    def action_effect(self, state, action):
        """Determines the correct and incorrect effect of actions"""

        if self.problem_type == 'gridworld':
            incorrect_actions = np.copy(self.enabled_actions[state])
            (s1,s2) = self.state_mapping[state]

            if self.enabled_actions[state,action]:
                if action == 0:
                    correct_state = state
                    incorrect_actions[[True, True, True, True, True]] = 0
                    return [correct_state, incorrect_actions]
                elif action == 1:
                    correct_state = (s1-1)*self.dim[1]+s2
                    incorrect_actions[[True, True, False, False, False]] = 0
                    return [correct_state, incorrect_actions]
                elif action == 2:
                    correct_state = s1*self.dim[1]+s2+1
                    incorrect_actions[[True, False, True, False, False]] = 0
                    return [correct_state, incorrect_actions]
                elif action == 3:
                    correct_state = (s1+1)*self.dim[1]+s2
                    incorrect_actions[[True, False, False, True, False]] = 0
                    return [correct_state, incorrect_actions]
                elif action == 4:
                    correct_state = s1*self.dim[1]+s2-1
                    incorrect_actions[[True, False, False, False, True]] = 0
                    return [correct_state, incorrect_actions]
                else:
                    raise NameError("Given action is not defined")
            else:
                return None
        else:
            raise NameError("Given problem type has no defined action effect")


    def gridworld_2D(self, dim=(24,37), p_correctmove=0.85, init_state=0):
        """Create an MDP for navigation in a 2D gridworld"""

        self.dim = dim # dimensions (d1, d2) of the state space
        self.states = np.arange(self.dim[0]*self.dim[1]) # set of states
        self.state_mapping = {i*self.dim[1]+j : (i,j)
                for i in range(self.dim[0])
                for j in range(self.dim[1])}
        self.init_state = init_state # initial state
        self.actions = np.arange(5) # set of actions
        self.action_mapping = {0 : 'stop', 1 : 'up', 2 : 'right', 3 : 'down', 4 : 'left'}
        self.enabled_actions = np.ones((len(self.states),len(self.actions)), dtype=np.bool) # enabled actions in states
        for state, coordinate in self.state_mapping.items():
            if coordinate[0] == 0: # top boundary
                self.enabled_actions[state,1] = 0
            elif coordinate[0] == self.dim[0]-1: # bottom boundary
                self.enabled_actions[state,3] = 0
            if coordinate[1] == 0: # left boundary
                self.enabled_actions[state,4] = 0
            elif coordinate[1] == self.dim[1]-1: # right boundary
                self.enabled_actions[state,2] = 0

        self.p_correctmove = p_correctmove # probability of correct execution of action
        self.transitions = np.zeros((len(self.states),len(self.actions),
                                     len(self.states)), dtype=np.float64) # transition function
        for state, coordinate in self.state_mapping.items():
            for action in self.actions[self.enabled_actions[state]]:
                [correct_state, incorrect_actions] = self.action_effect(state, action)
                n_inc_actions = np.sum(incorrect_actions)
                if n_inc_actions == 0:
                    self.transitions[state, action, correct_state] = 1
                else:
                    self.transitions[state, action, correct_state] = self.p_correctmove
                    for inc_action in self.actions[incorrect_actions]:
                        inc_state = self.action_effect(state, inc_action)[0]
                        self.transitions[state, action, inc_state] = (1-self.p_correctmove)/n_inc_actions

    def semantic_representation(self, property_dist='random', prior_belief='random'):
        """Assign semantics to MDP states"""

        if self.problem_type == 'simple':
            self.properties = np.arange(2)
            self.property_mapping = {0 : 'p', 1 : 'q'}
            self.label_true = np.array([[0,0],[0,0],[1,0]],
                                       dtype=np.bool) # true property labels of states
            self.label_belief = np.array([[0,0],[0,0.7],[0.6,0]],
                                         dtype=np.float64) # truthness confidence (belief) over property labels

        elif self.problem_type == 'book':
            self.properties = np.arange(1)
            self.property_mapping = {0 : 'target'}
            self.label_true = np.zeros((len(self.states),len(self.properties)),
                                       dtype=np.bool) # true property labels of states
            self.label_true[3,0] = True
            self.label_belief = np.zeros((len(self.states),len(self.properties)),
                                         dtype=np.float64) # truthness confidence (belief) over property labels
            self.label_belief[3,0] = 0.8; self.label_belief[6,0] = 0.25;

        elif self.problem_type == 'gridworld':
            self.properties = np.arange(2)
            self.property_mapping = {0 : 'obstacle', 1 : 'target'}

            self.label_true = np.zeros((len(self.states),len(self.properties)),
                                       dtype=np.bool) # true property labels of states
            if property_dist == 'random':
#                # random obstacles
#                n_obstacle = 10
#                obstacle_pos = np.random.randint(0,len(self.states),n_obstacle)
                obstacle_pos = [65,66,67,68,69,70,
                                102,103,104,105,106,107,
                                139,140,141,142,143,144,
                                160,161,162,163,164,165,166,167,168,169,170,171,172,176,177,178,179,180,181,
                                197,198,199,200,201,202,203,204,205,206,207,208,209,213,214,215,216,217,218,
                                234,235,236,237,238,239,240,241,242,243,244,245,246,250,251,252,253,254,255,
                                271,272,273,274,275,276,277,278,279,280,281,282,283,
                                311,312,313,314,315,316,317,
                                348,349,350,351,352,353,354,
                                376,377,378,379,380,381,382,385,386,387,388,389,390,391,401,402,
                                413,414,415,416,417,418,419,422,423,424,425,426,427,428,438,439,
                                450,451,452,453,454,455,456,473,474,475,476,477,
                                487,488,489,490,491,492,493,510,511,512,513,514,
                                524,525,526,527,528,529,530,
                                561,562,563,564,565,566,567,591,
                                598,599,600,601,602,603,604,628,
                                635,636,637,638,639,640,641,649,650,651,652,653,654,655,656,657,658,659,660,661,665,
                                672,673,674,675,676,677,678,686,687,688,689,690,691,692,693,694,695,696,697,698,702,
                                709,710,711,712,713,714,715,723,724,725,726,727,728,729,730,731,732,733,734,735,
                                746,747,748,749,750,751,752,760,761,762,763,764,765,766,767,768,769,770,771,772,
                                783,784,785,786,787,788,789,797,798,799,800,801,802,803,804,805,806,807,808,809,
                                820,821,822,823,824,825,826,834,835,836,837,838,839,840,841,842,843,844,845,846,
                                871,872,873,874,875,876,877,878,879,880,881,882,883]
                self.label_true[obstacle_pos,0] = 1
#                # random targets
#                n_target = 2
#                target_pos = np.random.randint(0,len(self.states),n_target)
                target_pos = [467,468,469,470,
                              504,505,506,507,
                              541,542,543,544,
                              578,579,580,581]
                self.label_true[target_pos,1] = 1

            self.label_belief = np.zeros((len(self.states),len(self.properties)),
                                         dtype=np.float64) # truthness confidence (belief) over property labels
            if prior_belief == 'exact':
                self.label_belief[:,:] = self.label_true

            elif prior_belief == 'random':
                self.label_belief = 0.5 * np.ones((len(self.states),len(self.properties)),
                                                  dtype=np.float64)

            elif prior_belief == 'noisy-ind':
                noise = 0.25
                for state in self.states:
                    for prop in self.properties:
                        if self.label_true[state,prop]:
                            self.label_belief[state,prop] = 1 - noise
                        else:
                            self.label_belief[state,prop] = noise

            elif prior_belief == 'noisy-dep':
                noise = 0.24
                confusion_level = 1
                for state in self.states:
                    for prop in self.properties:
                        if self.label_true[state,prop]:
                            self.label_belief[state,prop] = 1 - noise
                            neighbors = self.state_neighbor(state, confusion_level)
                            neighbors.remove(state)
                            self.label_belief[[True if s in neighbors
                                               else False
                                               for s in self.states],
                                              prop] = noise/len(neighbors)

            else:
                raise NameError("Given prior belief is not defined")

        else:
            raise NameError("Given problem type has no defined semantics")

    def state_neighbor(self, state, degree):
        """Find neighbors of a state up tp a given degree of closeness"""

        if self.problem_type == 'gridworld':
            neighbors = {state}
            checked = set()
            for d in range(degree):
                for n in set.difference(neighbors,checked):
                    for a in self.actions[self.enabled_actions[n]]:
                        new_n = self.action_effect(n,a)[0]
                        neighbors.add(new_n)
                    checked.add(new_n)
        else:
            raise NameError("Given problem type has no defined neighbors")

        return neighbors

    def make_absorbing(self, state):
        """Make a state of the MDP absorbing"""

        if self.problem_type == 'simple':
            self.enabled_actions[state] = [0,0,1] # only action "c" is enabled
            self.transitions[state,:,:] = np.zeros((len(self.actions),len(self.states))) # remove all transitions
            self.transitions[state,2,state] = 1 # add transition for "c"
        elif self.problem_type == 'book':
            self.enabled_actions[state] = [1,0] # only action "alpha" is enabled
            self.transitions[state,:,:] = np.zeros((len(self.actions),len(self.states))) # remove all transitions
            self.transitions[state,0,state] = 1 # add transition for "alpha"
        elif self.problem_type == 'gridworld':
            self.enabled_actions[state] = [1,0,0,0,0] # only action "stop" is enabled
            self.transitions[state,:,:] = np.zeros((len(self.actions),len(self.states))) # remove all transitions
            self.transitions[state,0,state] = 1 # add transition for "stop"
        else:
            raise NameError("Given problem type does not support absorbing states")

class Automaton:
    """Generate an automaton object for a specification"""

    def __init__(self, problem_type, spec=None):

        self.problem_type = problem_type

        if self.problem_type == 'reach_avoid':
            self.reach_avoid()
        else:
            raise NameError("Given problem type is not supported")

    def reach_avoid(self):
        """Generate a complete 3-state DFA for reach-avoid task"""

        self.states = np.arange(3) # set of states
        self.init_state = [0] # initial state
        self.alphabet = {0 : 'obstacle', 1 : 'target'} # transition alphabet
        self.actions = [(0,),(1,)] # actions from alphabet
        for l in range(len(self.alphabet)-1):
            self.actions = [a + (0,) for a in self.actions] + [a + (1,) for a in self.actions]
        self.transitions = np.array([
                [[1,0,0],[0,0,1],[0,1,0],[0,1,0]],
                [[0,1,0],[0,1,0],[0,1,0],[0,1,0]],
                [[0,0,1],[0,0,1],[0,0,1],[0,0,1]]], dtype=np.bool) # transition function
        self.acc_state = [1] # accepting states
        self.rej_state = [2] # rejecting states

def verifier(model, spec):
    """Find a policy that maximizes the probability of specifiation realization"""

    if type(model) == MC:
        model_type = 'MC'
    elif type(model) == MDP:
        model_type = 'MDP'
    else:
        raise NameError("Given model is not supported for verification")

    if len(spec) == 2:
        # reach-avoid specification
        if len(set.intersection(set(spec[0]),set(spec[1]))) != 0:
            print("The specification creates conflict --- reach set updated by removing avoid elements")
            spec[1] = list(set.difference(set(spec[1]),set(spec[0])))

        for b in spec[0]:
            model.make_absorbing(b)

        # define linear program model
        grb_model = grb.Model(name=model_type+" LP")
        grb_model.setParam('OutputFlag', False )

        # define variables
        n_vars = len(model.states)
        xs_vars = {i : grb_model.addVar(vtype=grb.GRB.CONTINUOUS,
                        lb=0,
                        ub=1,
                        name="x_{0}".format(i))
                for i in range(n_vars)}

        # define constraints
        constraints = dict()
        n_consts = 0

        constraints.update({j : grb_model.addConstr(
                        lhs=xs_vars[spec[1][j]],
                        sense=grb.GRB.EQUAL,
                        rhs=1,
                        name="constraint_{0}".format(j))
                for j in range(len(spec[1]))})
        n_consts += len(spec[1])

        for s in model.states:
            if s not in spec[0]:

                if type(model) == MC:
                    constraints.update({n_consts : grb_model.addConstr(
                                    lhs=xs_vars[s],
                                    sense=grb.GRB.GREATER_EQUAL,
                                    rhs=grb.quicksum([model.transitions[s,j]*xs_vars[j]
                                            for j in model.states]),
                                    name="constraint_{0}".format(n_consts))})
                    n_consts += 1

                elif type(model) == MDP:
                    for a in model.actions[model.enabled_actions[s]]:
                        constraints.update({n_consts : grb_model.addConstr(
                                        lhs=xs_vars[s],
                                        sense=grb.GRB.GREATER_EQUAL,
                                        rhs=grb.quicksum([model.transitions[s,a,j]*xs_vars[j]
                                                for j in model.states if model.transitions[s,a,j]!=0]),
                                        name="constraint_{0}".format(n_consts))})
                        n_consts += 1

        # define objective
        objective = grb.quicksum([xs_vars[i] for i in range(n_vars)])

        # add objective
        grb_model.ModelSense = grb.GRB.MINIMIZE
        grb_model.setObjective(objective)

        # solve
        grb_model.write(model_type+".lp")
        grb_model.optimize()

        # store data
        if grb_model.status == grb.GRB.status.OPTIMAL:
            grb_model.write(model_type+".sol")
        opt_df = pd.DataFrame.from_dict(xs_vars, orient="index",
                                        columns = ["variable_object"])
        opt_df.reset_index(inplace=True)
        opt_df["solution_value"] = opt_df["variable_object"].apply(lambda item: item.X)

    else:
        raise NameError("Given specification is not handled")

    vars_val = [v.x for v in grb_model.getVars()]

    # find the optimal policy
    opt_policy = None
    if type(model) == MDP:
        opt_policy = []

        for s in model.states:
            n_opt_act = 0
            opt_act = []
            for a in model.actions[model.enabled_actions[s]]:
                if isclose(vars_val[s],
                           np.sum([model.transitions[s,a,ss]*vars_val[ss]
                                   for ss in model.states if model.transitions[s,a,ss]!=0]),
                           abs_tol=1e-24):
                        opt_act.append(a)
                        n_opt_act += 1

            opt_policy.append(opt_act)

    return (vars_val, opt_policy)

def stat_verifier(model,s_current,policy,spec,n_sample):
    """Verify an induced Markov chain with sampling from uncertain properties"""

    pol_map = {}
    for s in model.states:
        opt_act = policy[s]
        if 0 in opt_act and len(opt_act)>1:
            opt_act = opt_act[1:]
        opt_act = np.random.choice(opt_act)
        pol_map[s] = opt_act
    policy = Policy(model, randomization=False, memory_use=False, mapping=pol_map)
    prob_sat = 0.0

    for sample in range(n_sample):

        avoid_set = []
        reach_set = []
        for s in model.states:
            if np.random.binomial(1, model.label_belief[s,0]):
                avoid_set.append(s)

            if np.random.binomial(1, model.label_belief[s,1]):
                reach_set.append(s)

        samp_spec=[avoid_set,reach_set]
        pr_sample = policy.evaluate(samp_spec,s_current)
        prob_sat += pr_sample

    prob_sat = prob_sat/n_sample

    return prob_sat

def obs_modeling(model):
    """Model the observations"""

    p_obs_model = np.zeros((len(model.states),len(model.properties)),dtype=np.float64)
    for s in range(len(model.states)):
        for p in range(len(model.properties)):
            p_obs_model[s,p] = np.random.uniform(0.6, 0.95)

    obs = np.zeros((len(model.states),len(model.properties)),dtype=np.bool)
    for s in range(len(model.states)):
        for p in range(len(model.properties)):
            if model.label_true[s,p]:
                obs[s,p] = np.random.binomial(1, p_obs_model[s,p], 1)
            else:
                obs[s,p] = np.random.binomial(1, 1.0-p_obs_model[s,p], 1)

    return (obs,p_obs_model)

def belief_update(label_belief, obs, p_obs_model, bayes_flag):
    """Update the belief over properties"""

    if bayes_flag:
        posterior_belief = np.copy(label_belief)
        for i_s,s in enumerate(label_belief):
            for i_p,p in enumerate(s):
                if obs[i_s,i_p]:
                    posterior_belief[i_s,i_p] = label_belief[i_s,i_p]*p_obs_model[i_s,i_p] /\
                        (label_belief[i_s,i_p]*p_obs_model[i_s,i_p] +
                         (1-label_belief[i_s,i_p])*(1-p_obs_model[i_s,i_p]))
                else:
                    posterior_belief[i_s,i_p] = label_belief[i_s,i_p]*(1-p_obs_model[i_s,i_p]) /\
                        (label_belief[i_s,i_p]*(1-p_obs_model[i_s,i_p]) +
                         (1-label_belief[i_s,i_p])*p_obs_model[i_s,i_p])
    else:
        posterior_belief = np.copy(label_belief)
        for i_s,s in enumerate(label_belief):
            for i_p,p in enumerate(s):
                if obs[i_s,i_p]:
                    posterior_belief[i_s,i_p] = p_obs_model[i_s,i_p]
                else:
                    posterior_belief[i_s,i_p] = 1.0-p_obs_model[i_s,i_p]

    return posterior_belief

def info_qual(label_belief):
    """Compute quality of current information with entropy measure"""

    # entropy of state = sum of entropy of its properties
    # entropy of belief = average of entropy of states

    ent = np.zeros(label_belief.shape)
    for i_s,s in enumerate(label_belief):
        for i_p,p in enumerate(s):
            p_ber = label_belief[i_s,i_p]

            if p_ber == 0 or p_ber == 1:
                pass
            else:
                p_ber = label_belief[i_s,i_p]
                ent[i_s,i_p] = - (p_ber*np.log(p_ber) +
                                  (1-p_ber)*np.log(1-p_ber))

    ent = np.mean(ent)

    return ent

def jsd(x,y): # Jensen-shannon divergence
    import warnings
    warnings.filterwarnings("ignore", category = RuntimeWarning)
    x = np.array(x)
    y = np.array(y)
    d1 = x*np.log(2*x/(x+y))
    d2 = y*np.log(2*y/(x+y))
    d1[np.isnan(d1)] = 0
    d2[np.isnan(d2)] = 0
    d = 0.5*np.sum(d1+d2)
    return d

def info_div(pr_belief, post_belief):
    """Compute divergence between prior and posterior belief
    with Jensen–Shannon divergence"""

    # total divergence = sum of divergence for all states and properties

    div = np.zeros(pr_belief.shape)
    for i_s,s in enumerate(pr_belief):
        for i_p,p in enumerate(s):
            pr_ber = pr_belief[i_s,i_p]
            post_ber = post_belief[i_s,i_p]
            div[i_s,i_p] = jsd([pr_ber,1-pr_ber],[post_ber,1-post_ber])

    div = np.mean(div)

    return div

def estimate_AP(label_dist, method='mode'):
    """Estimate the truth value of atomic propositions"""

    label_est = np.zeros(label_dist.shape,dtype=np.bool) # estimated property labels of states

    if method == 'mode':
        for i_s,s in enumerate(label_est):
            for i_p,p in enumerate(s):
                if label_dist[i_s,i_p] > 0.5:
                    label_est[i_s,i_p] = True
                else:
                    label_est[i_s,i_p] = False

    elif method == 'risk-averse':
        thresh_good = 0.5 # higher --> less willing to go to a potential good state
        thresh_bad = 0.3 # higher --> more willing to go to a potential bad state

        for i_s,s in enumerate(label_est):
            # needs adjustment for specifications beyond reach-avoid
            if label_dist[i_s,0] > thresh_bad:
                label_est[i_s,0] = True
            else:
                label_est[i_s,0] = False

            if label_dist[i_s,1] > thresh_good:
                label_est[i_s,1] = True
            else:
                label_est[i_s,1] = False

    return label_est

def simulation(model, spec, perc_flag=False, bayes_flag=False,
               replan_flag=False, div_test_flag=False, act_info_flag=False):
    """Simulate passive perception and online replanning"""
    # spec not used

    # simulation results
    term_flag = False
    task_flag = False
    timestep = 0
    max_timestep = 50
    plan_count = 0
    div_thresh = 0.001
    n_sample = 10
    risk_thresh = 0.5
    state_hist = []
    state_hist.append(model.init_state)
    action_hist = [[],[]] # [[chosen action],[taken action]]
    infqual_hist = []
    infqual_hist.append(info_qual(model.label_belief))
    risk_hist = []

    while not term_flag:

        if perc_flag:
            # estimate map
            label_est = estimate_AP(model.label_belief, method='risk-averse')
            spec_est = [[],[]]
            for s in range(len(model.states)):
                if label_est[s,0] == True:
                    spec_est[0].append(s)
                if label_est[s,1] == True:
                    spec_est[1].append(s)

        if replan_flag or (not replan_flag and plan_count==0):
            # find an optimal policy
            (vars_val, opt_policy) = verifier(copy.deepcopy(model), spec_est)
            plan_count += 1

        if act_info_flag:
            # risk evaluation
            prob_sat = stat_verifier(model,state_hist[-1],opt_policy,spec,n_sample)
            risk = np.abs(vars_val[state_hist[-1]] - prob_sat); print(vars_val[state_hist[-1]],prob_sat)
            risk_hist.append(risk)
            print("Risk due to Perception Uncertainty:   ",risk)

            # perception policy
            if risk > risk_thresh:
                # implement perception policy
                timestep += 1
                state = state_hist[-1]
                action = 0
            else:
                pass
        timestep += 1
        print("Timestep:   ",timestep)
        state = state_hist[-1]
        opt_act = opt_policy[state]
        if 0 in opt_act and len(opt_act)>1:
            opt_act = opt_act[1:]

        action = np.random.choice(opt_act)

        action_hist[0].append(action)
        next_state = np.random.choice(model.states, p=model.transitions[state,action])
        # identify taken action
        for a in model.actions[model.enabled_actions[state]]:
            if model.action_effect(state,a)[0] == next_state:
                action_taken = a
        action_hist[1].append(action_taken)
        state_hist.append(next_state)

        # get new information
        (obs,p_obs_model) = obs_modeling(model)

        # update belief
        next_label_belief = belief_update(model.label_belief, obs,
                                          p_obs_model, bayes_flag)
        # divergence test on belief
        if div_test_flag:
            div = info_div(model.label_belief,next_label_belief)
            print("Belief Divergence:   ",div)
            if info_div(model.label_belief,next_label_belief) > div_thresh:
                replan_flag = True
            else:
                replan_flag = False
        model.label_belief = np.copy(next_label_belief)
        infqual_hist.append(info_qual(model.label_belief))

        # check task realization
        if model.label_true[state_hist[-1],0] == True:
            term_flag = True
            print("at a state with an obstacle")

        if model.label_true[state_hist[-1],1] == True:
            task_flag = True
            term_flag = True
            print("at a target state")

        if timestep == max_timestep:
            term_flag = True
            print("timestep exceeded the maximum limit")

    print("Number of Time Steps:   ",timestep)
    print("Number of Replanning:   ",plan_count)

    return (state_hist,action_hist,infqual_hist,risk_hist,timestep,plan_count,task_flag)

if __name__ == '__main__':

    # define simulation parameters
    n_iter = 20
    infqual_hist_all = []
    risk_hist_all = []
    timestep_all = []
    plan_count_all = []
    task_flag_all = []

    for iter in range(n_iter):

        # create problem setting
        model = MDP('gridworld')
        model.semantic_representation(prior_belief='random')
        perc_flag = True
        bayes_flag = True
        replan_flag = True
        div_test_flag = True
        act_info_flag = True
        spec_true = [[],[]]
        for s in range(len(model.states)):
            if model.label_true[s,0] == True:
                spec_true[0].append(s)
            if model.label_true[s,1] == True:
                spec_true[1].append(s)

        # simulate: plan, exploit
        (state_hist,action_hist,infqual_hist,risk_hist,timestep,plan_count,task_flag) = simulation(
                model,spec_true,
                perc_flag,bayes_flag,replan_flag,div_test_flag,act_info_flag)

        infqual_hist_all.append(infqual_hist)
        risk_hist_all.append(risk_hist)
        timestep_all.append(timestep)
        plan_count_all.append(plan_count)
        task_flag_all.append(int(task_flag))

    task_rate = np.mean(task_flag_all)
