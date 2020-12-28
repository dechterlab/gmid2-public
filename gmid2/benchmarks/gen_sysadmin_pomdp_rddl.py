"""
same as sysadmin_mdp

each state variable emits observations, P(obs | running)
        running true Bern( 0.95)    false Bern(0.05)
        constraint changes
        that's all!
"""
import os
import random
import networkx as nx
from gmid2.global_constants import *
from gmid2.basics.uai_files import *
from gmid2.benchmarks.gen_sysadmin_mdp_rddl import *


def create_sysadmin_obs(obs_prob):
    return [obs_prob, 1-obs_prob, 1-obs_prob, obs_prob]

def make_2t_sysadmin_pomdp_e2(n_s, graph_connectivity, init_state_list: List[bool], reboot_prob=0.1, reboot_penalty=0.75,
                            constraint_violation_cost=100, obs_prob=0.95):
    constraint_size = max(2, 2 + max((len(graph_connectivity[k]) for k in graph_connectivity)))
    pomdp_2t = nx.DiGraph()

    # init states (same as mdp)
    node_label = 0
    for n in range(n_s):        # 0 ~ n_s-1
        pomdp_2t.add_node(n)
        pomdp_2t.nodes[n]['type'] = 'initial_state'
        pomdp_2t.nodes[n]['domain_size'] = 2
        pomdp_2t.nodes[n]['parents'] = []  # empty
        pomdp_2t.nodes[n]['table'] = create_sysadmin_init(init_state=init_state_list[n])

    # observed states
    node_label += n_s
    for i, n in enumerate(range(node_label, node_label + n_s)):     #n_s ~ 2*n_s - 1
        pomdp_2t.add_node(n)
        pomdp_2t.nodes[n]['type'] = 'obs_state'
        pomdp_2t.nodes[n]['domain_size'] = 2
        pomdp_2t.nodes[n]['parents'] = [i]
        pomdp_2t.nodes[n]['table'] = create_sysadmin_obs(obs_prob)

    # decision variables
    node_label += n_s
    decision_labels = []
    for i, n in enumerate(range(node_label, node_label + n_s)):     # 2*n_s ~ 3*n_s - 1
        decision_labels.append(n)
        pomdp_2t.add_node(n)
        pomdp_2t.nodes[n]['type'] = 'decision'
        pomdp_2t.nodes[n]['domain_size'] = 2
        pomdp_2t.nodes[n]['parents'] = list(range(n_s, 2*n_s))  # observations are parents
        pomdp_2t.nodes[n]['table'] = []

    # transition states
    node_label += n_s
    for i, n in enumerate(range(node_label, node_label + n_s)):     # 3*n_s ~ 4*n_s - 1
        pomdp_2t.add_node(n)
        pomdp_2t.nodes[n]['type'] = 'state'
        pomdp_2t.nodes[n]['domain_size'] = 2
        pomdp_2t.nodes[n]['action_parents'] = [decision_labels[i]]
        pomdp_2t.nodes[n]['state_parents'] = [i] + graph_connectivity[i]
        pomdp_2t.nodes[n]['parents'] = pomdp_2t.nodes[n]['action_parents'] + pomdp_2t.nodes[n]['state_parents']
        pomdp_2t.nodes[n]['table'] = create_sysadmin_trans(len(graph_connectivity[i]), reboot_prob)  # don't sort parents

    # build constraint tree
    node_label += n_s
    root = constraint_tree(decision_labels, constraint_size)
    l = post_traversal(root)
    tree2dbn = {}
    for node in l:
        if node.action_label:
            tree2dbn[node.node_id] = node.action_label
    for n, node in enumerate((el for el in l if el.action_label is None), start=node_label):
        pomdp_2t.add_node(n)
        tree2dbn[node.node_id] = n  # node.node_id 0 for the root, 1,2,3... for actions and temp follows
        pomdp_2t.nodes[n]['type'] = 'constraint'
        pomdp_2t.nodes[n]['domain_size'] = 3  # domain 0 zero selection 1 one selection 2 more than one selection
        pomdp_2t.nodes[n]['parents'] = sorted([tree2dbn[k.node_id] for k in node.ch])
        pomdp_2t.nodes[n]['table'] = create_sysadmin_constraint_rec2(pomdp_2t.nodes[n]['parents'], decision_labels,
                                                                     node.node_id == 0)
    constraint_root_node = tree2dbn[root.node_id]  # this is now parent to all utility
    n_latent = len(l) - n_s

    # utility functions for states
    node_label += n_latent
    for i, n in enumerate(range(node_label, node_label + n_s)):
        pomdp_2t.add_node(n)
        pomdp_2t.nodes[n]['type'] = 'utility'
        pomdp_2t.nodes[n]['domain_size'] = None
        pomdp_2t.nodes[n]['state_parents'] = [n - n_s - n_latent]
        pomdp_2t.nodes[n]['parents'] = [constraint_root_node] + pomdp_2t.nodes[n]['state_parents']
        pomdp_2t.nodes[n]['table'] = [0.0, 1.0] + [0.0, 1.0] + [-constraint_violation_cost] * 2

    # utility cost for actions
    node_label += n_s
    for i, n in enumerate(range(node_label, node_label + n_s)):
        pomdp_2t.add_node(n)
        pomdp_2t.nodes[n]['type'] = 'utility'
        pomdp_2t.nodes[n]['domain_size'] = None
        pomdp_2t.nodes[n]['action_parents'] = [n - 3 * n_s - n_latent]
        pomdp_2t.nodes[n]['parents'] = [constraint_root_node] + pomdp_2t.nodes[n]['action_parents']
        pomdp_2t.nodes[n]['table'] = [0.0, -constraint_violation_cost, 0.0, -reboot_penalty,
                                      -constraint_violation_cost, -constraint_violation_cost]
    return pomdp_2t


def unroll_2t_sysadmin_pomdp_e2(pomdp_2t: nx.DiGraph, time_horizon, file_name):
    id_diagram = nx.DiGraph()

    initial_states = []
    obs_states = []
    decision_nodes = []
    utility_nodes = []
    state_nodes = []
    constraint_nodes = []
    for n in pomdp_2t:
        node_type = pomdp_2t.nodes[n]['type']
        if node_type == 'initial_state':
            initial_states.append(n)
        elif node_type == "obs_state":
            obs_states.append(n)
        elif node_type == 'decision':
            decision_nodes.append(n)
        elif node_type == 'state':
            state_nodes.append(n)
        elif node_type == 'utility':
            utility_nodes.append(n)
        elif node_type == 'constraint':
            constraint_nodes.append(n)
    initial_states = sorted(initial_states)
    obs_nodes = sorted(obs_states)
    utility_nodes = sorted(utility_nodes)
    state_nodes = sorted(state_nodes)
    constraint_nodes = sorted(constraint_nodes)
    n_s, n_o, n_a, n_u, n_c = len(state_nodes), len(obs_nodes), len(decision_nodes), len(utility_nodes), len(constraint_nodes)

    partial_ordering = []  # specific to this pomdp structure
    hidden_nodes = []
    domains = []
    id_node_to_pomdp_2t = {}  # influence diagram node to pomdp node
    id_node_to_var_id = {}  # influence diagram node to variable id in uai file

    # init state
    node_label, var_id = 0, 0
    for s_node in initial_states:
        id_diagram.add_node(node_label)
        hidden_nodes.append(node_label)
        id_node_to_var_id[node_label] = var_id
        id_node_to_pomdp_2t[node_label] = s_node
        domains.append(pomdp_2t.nodes[s_node]['domain_size'])
        id_diagram.nodes[node_label]['parents'] = []  # no parents for init state
        node_label += 1
        var_id += 1


    for th in range(time_horizon):
        one_stage_offset = (n_s + n_o + n_a + n_u + n_c) * th       # num nodes added after a stage
        state_offset = max(0, (n_s + n_a + n_o) + (th-1) * (n_s + n_o + n_a + n_u + n_c))

        # observation
        current_obs_nodes = []
        for o_node in obs_nodes:        # numbers in 2t DBN
            id_diagram.add_node(node_label)
            id_node_to_var_id[node_label] = var_id
            id_node_to_pomdp_2t[node_label] = o_node
            domains.append(pomdp_2t.nodes[o_node]['domain_size'])
            current_obs_nodes.append(node_label)
            id_diagram.nodes[node_label]['parents'] = [k + state_offset for k in pomdp_2t.nodes[o_node]['parents']]
            for pa in id_diagram.nodes[node_label]['parents']:
                id_diagram.add_edge(pa, node_label)
            node_label += 1
            var_id += 1
        partial_ordering.append(current_obs_nodes)

        # decision
        current_decision_nodes = []
        for d_node in decision_nodes:
            id_diagram.add_node(node_label)
            id_node_to_var_id[node_label] = var_id
            id_node_to_pomdp_2t[node_label] = d_node
            domains.append(pomdp_2t.nodes[d_node]['domain_size'])
            current_decision_nodes.append(node_label)
            id_diagram.nodes[node_label]['parents'] = [k + one_stage_offset for k in pomdp_2t.nodes[d_node]['parents']]
            for pa in id_diagram.nodes[node_label]['parents']:
                id_diagram.add_edge(pa, node_label)
            node_label += 1
            var_id += 1
        partial_ordering.append(current_decision_nodes)

        # states hidden
        for s_node in state_nodes:
            id_diagram.add_node(node_label)
            hidden_nodes.append(node_label)
            id_node_to_var_id[node_label] = var_id
            id_node_to_pomdp_2t[node_label] = s_node
            domains.append(pomdp_2t.nodes[s_node]['domain_size'])
            id_diagram.nodes[node_label]['parents'] = [k + one_stage_offset for k in
                                                       pomdp_2t.nodes[s_node]['action_parents']]
            id_diagram.nodes[node_label]['parents'] += [k + state_offset for k in
                                                        pomdp_2t.nodes[s_node]['state_parents']]
            for pa in id_diagram.nodes[node_label]['parents']:
                id_diagram.add_edge(pa, node_label)
            node_label += 1
            var_id += 1

        # constraints  hidden
        for c_node in constraint_nodes:
            id_diagram.add_node(node_label)
            hidden_nodes.append(node_label)
            id_node_to_var_id[node_label] = var_id
            id_node_to_pomdp_2t[node_label] = c_node
            domains.append(pomdp_2t.nodes[c_node]['domain_size'])
            id_diagram.nodes[node_label]['parents'] = [k + one_stage_offset for k in pomdp_2t.nodes[c_node]['parents']]
            for pa in id_diagram.nodes[node_label]['parents']:
                id_diagram.add_edge(pa, node_label)
            node_label += 1
            var_id += 1

        # utility
        for u_node in utility_nodes:
            id_diagram.add_node(node_label)
            id_node_to_pomdp_2t[node_label] = u_node
            id_diagram.nodes[node_label]['parents'] = [k + one_stage_offset for k in pomdp_2t.nodes[u_node]['parents']]
            for pa in id_diagram.nodes[node_label]['parents']:
                id_diagram.add_edge(pa, node_label)
            node_label += 1

    partial_ordering.append(hidden_nodes)
    write_id_for_fhpomdp(file_name, time_horizon, domains, partial_ordering,
                       n_s, n_o, n_a, n_u, n_c,
                       pomdp_2t, id_diagram, id_node_to_pomdp_2t, id_node_to_var_id)
    return id_diagram


def write_id_for_fhpomdp(file_name, time_horizon, domains, partial_ordering,
                         num_state_nodes, num_obs_nodes, num_decision_nodes, num_util_nodes, num_constraint_nodes,
                         pomdp_2t, id_diagram,
                         id_node_to_pomdp, id_node_to_var_id):
    f = open(file_name + '.uai', 'w')
    f.write('ID\n')
    # vars
    n_vars = sum([len(bk) for bk in partial_ordering] )
    f.write('{}\n'.format(n_vars))
    for d in domains:
        f.write('{} '.format(d))
    f.write('\n')
    # factors
    n_factors = num_state_nodes + (num_state_nodes + num_obs_nodes + num_util_nodes + num_constraint_nodes) * time_horizon
    f.write('{}\n'.format(n_factors))

    for n in sorted(iter(id_diagram)):
        if pomdp_2t.nodes[id_node_to_pomdp[n]]['type'] in ['initial_state', 'obs_state', 'state', 'constraint']:
            scope = id_diagram.nodes[n]['parents'] + [n]
        elif pomdp_2t.nodes[id_node_to_pomdp[n]]['type'] == 'utility':
            scope = id_diagram.nodes[n]['parents']
        elif pomdp_2t.nodes[id_node_to_pomdp[n]]['type'] == 'decision':
            continue
        else:
            assert False, "unknown node type"

        f.write('{} '.format(len(scope)))
        for s in scope:
            try:
                f.write('{} '.format(id_node_to_var_id[s]))
            except:
                assert False
        f.write('\n')
    f.write('\n')

    # tables
    for n in sorted(iter(id_diagram)):
        if pomdp_2t.nodes[id_node_to_pomdp[n]]['type'] != 'decision':
            f.write('{}\n'.format(len(pomdp_2t.nodes[id_node_to_pomdp[n]]['table'])))
            for t in pomdp_2t.nodes[id_node_to_pomdp[n]]['table']:
                f.write('{}\n'.format(float(t)))
            f.write('\n')
    f.close()

    f = open(file_name + '.id', 'w')
    f.write('{}\n'.format(n_vars))
    # var types
    for n in sorted(iter(id_diagram)):
        if pomdp_2t.nodes[id_node_to_pomdp[n]]['type'] in ['initial_state', 'obs_state', 'state', 'constraint']:
            f.write('C ')
        elif pomdp_2t.nodes[id_node_to_pomdp[n]]['type'] == 'decision':
            f.write('D ')
    f.write('\n')
    # factor types
    f.write('{}\n'.format(n_factors))
    for n in sorted(iter(id_diagram)):
        if pomdp_2t.nodes[id_node_to_pomdp[n]]['type'] in ['initial_state', 'obs_state', 'state', 'constraint']:
            f.write('P ')
        elif pomdp_2t.nodes[id_node_to_pomdp[n]]['type'] == 'utility':
            f.write('U ')
    f.write('\n')
    f.close()

    f = open(file_name + '.pvo', 'w')
    f.write('{};\n'.format(n_vars))
    f.write('{};\n'.format(len(partial_ordering)))

    for block in reversed(partial_ordering):
        for b in block:
            f.write('{} '.format(id_node_to_var_id[b]))
        f.write(';\n')
    f.close()


if __name__ == "__main__":
    config_num = 10
    g = {
        0: [6, 21, 23, 32],
        1: [3, 10, 11, 18, 36, 37, 39],
        2: [24],
        3: [],
        4: [],
        5: [13, 15, 23, 31],
        6: [0, 16, 21, 26],
        7: [19, 45],
        8: [5, 12, 16, 18],
        9: [],
        10: [],
        11: [0, 26, 35, 42],
        12: [48],
        13: [],
        14: [18, 24, 25, 31, 46],
        15: [38, 47],
        16: [6, 10, 34, 36, 41],
        17: [6, 21, 29, 30, 46],
        18: [49],
        19: [3, 7, 37, 49],
        20: [2, 26, 41],
        21: [],
        22: [8, 37, 45],
        23: [22, 33, 39],
        24: [14, 35, 44],
        25: [27, 28, 29, 42],
        26: [1, 11, 15, 41],
        27: [1, 34, 39],
        28: [33, 42, 44],
        29: [10, 16, 17],
        30: [9, 13, 15, 31],
        31: [49],
        32: [24],
        33: [1, 2, 5, 14, 20, 35],
        34: [5, 19],
        35: [4, 11, 13, 20, 25, 33, 34],
        36: [9],
        37: [12, 17, 40, 47],
        38: [7, 12, 22, 46],
        39: [0, 3, 32, 40, 45, 48],
        40: [48],
        41: [2, 29, 30],
        42: [20, 27, 38],
        43: [4, 9, 22, 28, 30, 32],
        44: [4, 47],
        45: [23, 25, 38, 44],
        46: [7, 8, 40],
        47: [14, 27],
        48: [8, 43],
        49: [17],
    }
    n_servers = 50
    reboot_prob = 0.0020
    reboot_penalty = 0.1
    obs_prob = 0.95

    constraint_violation_cost = 100
    for time_horizon in range(3, 11):
        problem_path = os.path.join(BENCHMARK_DIR, "sysadmin_pomdp")
        problem_name = "_".join(["sysadmin_inst_pomdp", str(config_num), "s="+str(n_servers),"t="+str(time_horizon)])
        file_name = os.path.join(problem_path, problem_name)

        pomdp_2t = make_2t_sysadmin_pomdp_e2(n_servers, g, [True] * n_servers, reboot_prob, reboot_penalty,
                                             constraint_violation_cost)
        unroll_2t_sysadmin_pomdp_e2(pomdp_2t, time_horizon=time_horizon, file_name=file_name)
