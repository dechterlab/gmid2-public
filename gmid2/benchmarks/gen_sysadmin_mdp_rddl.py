"""
Influence diagram for SysAdmin Domain from RDDL.
[SysAdminRddl](https://github.com/ssanner/rddlsim/tree/master/files/final_comp/rddl)

* SysAdmin problem re-written in RDDL language
* This script generates equivalent influence diagrams
  * inputs:
    * is_mdp = true if problem is MDP
    * number of servers
    * connectivity by a directed graph (ci, cj)
      * directed graph because CONNECTED(c1, c2) does not guarantee CONNECTED(c2, c1) in predicates
    * max number of neighbors (typically less than 5)
    * reboot_prob = 0.05
    * observ_prob = 0.95
    * reboot_prob = 0.1 (default)
    * reboot_penalty = 0.75 (default)
    * number of concurrent actions (constraints over actions)
      * max_scope_constraint
      * table constraint
    * init state of servers
      * 0 for off 1 for on


* overview of encodin
state variables are fixed to n_servers
build 2t-mdp and unroll it
init state, decision, state transition is obvious
utility per state and decision is also obvious

for the constraint, create a tree with a root (latent variable) having all decisions as its children
from maximum number of branching factor, recursively add intermediate latent variables under the root
the last leaf layer corresponds to decision nodes
tree2dbn dict maps the id of the tree (created from 0 using internal counter) to the node id in mdp
initally decision variables have node id because they are decision nodes
iterate each latent variables, add it to mdp, map the tree id and dbn id,
the node is visited in post traversal order so it always visit lower layer first.
at the point of connecting the parent node in mdp (children in tree), the node exists
attach table for each constraint

* input example
    # n_servers = 3
    # g = {
    #     0: [1],
    #     1: [],
    #     2: [1]
    # }
    # transition
    # action | current | n1, n2, ..., n | next |
    # CONNECTED(y, x)  first index is summed in expression
    # connectivity g = { x: [y1, y2, ... ] } list runs over the first servers on y
    # one utility per decision
    # constraint connecting all decisions, if too many make a tree intermediate latent variables
"""
import os
import random
import networkx as nx
from gmid2.global_constants import *
from gmid2.basics.uai_files import *


# def random_connectivity_undirected_graph(n_s, min_neighbors=2, max_neighbors=3):
#     """
#     :param n_s: total number of servers
#     :param min_neighbors: minimum nunmber of connections at least 2 to make a ring
#     :param max_neighbors: largest number of connections
#     :return: a random graph of servers
#     """
#     g = {k: [] for k in range(n_s)}
#     all_nodes = list(range(n_s))
#     for n in g:
#         num_current_edges = len( g[n] )
#         if num_current_edges >= max_neighbors:
#             continue
#         new_random_edges = random.randint(max(0, min_neighbors - num_current_edges), max_neighbors - num_current_edges)
#         if new_random_edges == 0:
#             continue
#
#         destinations = []
#         random.shuffle(all_nodes)
#         for dest in all_nodes:
#             if dest != n and dest not in destinations and dest not in g[n] and len(g[dest]) < max_neighbors:
#                 destinations.append(dest)
#             if len(destinations) == new_random_edges:
#                 break
#
#         g[n] += destinations
#
#         for dest in destinations:
#             if n not in g[dest]:
#                 g[dest].append(n)       # undirected graph connect in both ways
#     for n in g:
#         g[n] = sorted(g[n])
#     return g


def random_connectivity_graph(n_s, min_neighbors=2, max_neighbors=3):
    """
    :param n_s: total number of servers
    :param min_neighbors: minimum nunmber of connections at least 2 to make a ring
    :param max_neighbors: largest number of connections
    :return: a random directed graph (possibly with cycles) of servers
    """
    g = {k: [] for k in range(n_s)}
    all_nodes = list(range(n_s))
    for n in g:
        n_edges = random.randint(min_neighbors, max_neighbors)
        dest = []
        random.shuffle(all_nodes)
        for el in all_nodes:
            if el not in dest and el != n:
                dest.append(el)
            if len(dest) == n_edges:
                break
        g[n] = dest

    for n in g:
        g[n] = sorted(g[n])
    return g


def create_sysadmin_init(init_state=True):
    return [0.0, 1.0] if init_state else [1.0, 0.0]


def count_one_bits(k):
    cnt = 0
    while k:
        cnt += k & 1
        k >>= 1
    return cnt


def create_sysadmin_trans(num_neighbors, reboot_prob):
    """ scope of the table looks like
        [ action | prev state | nhd1 state | next state ]
        UAI convention is enum prob following the configurations
        0000, 0001, 0002, ... 1111
        first half is for action 0
        first half of the first half is for action  0 and prev state 0

        action off
            prev state off          -> reboot with reboot prob whatever other states are
                all nhd combinations
                turn on with reboot probability     [1-reboot, reboot]

            prev state on           -> if on then formula
                all nhd combinations
                turn on with prob 0.45 + 0.5* [1+ neighbor_on] / [1+all neighbors]

        action on
            all (nhd+1) combinations
            turn on with prob 1.0   [0.0, 1.0]
    """
    action_off_state_off = [1.0 - reboot_prob, reboot_prob]* pow(2, num_neighbors)

    action_off_state_on = []
    for k in range(pow(2, num_neighbors)):
        computer_on = count_one_bits(k)
        prob_on = 0.45 + 0.5* (1 + computer_on) / (1 + num_neighbors)
        action_off_state_on.append(1- prob_on)
        action_off_state_on.append(prob_on)

    action_on = [0.0, 1.0]* pow(2, num_neighbors+1)

    return action_off_state_off + action_off_state_on + action_on


def create_sysadmin_constraint_rec(parent_labels, decision_labels, is_root=False):
    # action can be no-op, 1-selection (if more than 1 selection allowed change domains size, this function)
    table = []
    def helper(pa_ind=0, bit_sum=0):
        if pa_ind == len(parent_labels):
            if bit_sum == 0:
                # table.append(1.0 if not is_root else 0.0)
                table.append(1.0)   # root can have 1.0 -> no-op is valid option
                table.append(0.0)
                table.append(0.0)
            elif bit_sum == 1:
                table.append(0.0)
                table.append(1.0)
                table.append(0.0)
            else:
                table.append(0.0)
                table.append(0.0)
                table.append(1.0 if not is_root else 0.0)   # more than 2 selection is invalid
                # table.append(1.0)  # root can have 1.0 -> no-op is valid option
            return

        if parent_labels[pa_ind] in decision_labels:
            helper(pa_ind + 1, bit_sum)
            helper(pa_ind + 1, bit_sum+1)
        else:
            helper(pa_ind + 1, bit_sum)
            helper(pa_ind + 1, bit_sum + 1)
            helper(pa_ind + 1, bit_sum + 2)     # temp adding 2 is any selection more than 2
    helper()
    return table

def create_sysadmin_constraint_rec2(parent_labels, decision_labels, is_root=False):
    # action can be no-op, 1-selection (if more than 1 selection allowed change domains size, this function)
    table = []
    def helper(pa_ind=0, bit_sum=0):
        if pa_ind == len(parent_labels):
            if bit_sum == 0:
                # table.append(1.0 if not is_root else 0.0)
                table.append(1.0)   # root can have 1.0 -> no-op is valid option
                table.append(0.0)
                table.append(0.0)
            elif bit_sum == 1:
                table.append(0.0)
                table.append(1.0)
                table.append(0.0)
            else:
                table.append(0.0)
                table.append(0.0)
                # table.append(1.0 if not is_root else 0.0)   # more than 2 selection is invalid
                table.append(1.0)  # root can have 1.0 -> no-op is valid option
            return

        if parent_labels[pa_ind] in decision_labels:
            helper(pa_ind + 1, bit_sum)
            helper(pa_ind + 1, bit_sum+1)
        else:
            helper(pa_ind + 1, bit_sum)
            helper(pa_ind + 1, bit_sum + 1)
            helper(pa_ind + 1, bit_sum + 2)     # temp adding 2 is any selection more than 2
    helper()
    return table


# def create_sysadmin_constraint(num_parents):
#     """
#     for each configuration of parents, add 2 rows [0.0, 0.0 or 1.0 depending on config]
#         if config only contain 1 bit then 1.0 otherwise 0.0
#     """
#     table = []
#     for i in range(pow(2, num_parents)):
#
#         if count_one_bits(i) == 1:
#             table.append(0.0)
#             table.append(1.0)
#         else:
#             table.append(0.0)
#             table.append(0.0)
#     return table


class TreeNode():
    count=0
    def __init__(self, server=None, action_label=None):
        self.node_id = TreeNode.count
        TreeNode.count += 1
        self.server = server
        self.action_label = action_label
        self.pa = []
        self.ch = []

    def __repr__(self):
        return "node:{},server:{}".format(self.node_id, self.server)


def post_traversal(root):
    def helper(root):
        for ch in root.ch:
            helper(ch)
        res.append(root)
    res = []
    helper(root)
    return res


def constraint_tree(decision_labels, constraint_size):
    def partition_children(flat):
        partitioned = []
        partition = []
        i = 0
        while i < len(flat):
            if len(partition) == constraint_size:
                partitioned.append(partition)
                partition = []
            partition.append( flat[i])
            i += 1
        if partition:
            partitioned.append( partition)
        return partitioned

    root = TreeNode()
    for i, n in enumerate(decision_labels):
        t = TreeNode(server=i, action_label=n)
        root.ch.append(t)
        t.pa.append(root)

    while len(root.ch) > constraint_size:
        partitions = partition_children( random.sample(root.ch, len(root.ch)) )
        root.ch = []
        for p in partitions:
            if len(p) > 1:
                t = TreeNode()
                root.ch.append(t)
                t.pa = root
                t.ch = p
                for t2 in p:
                    t2.pa = t
            else:
                root.ch.append(p[0])
    return root


def make_2t_sysadmin_mdp_e1(n_s, graph_connectivity, init_state_list:List[bool], reboot_prob=0.1, reboot_penalty=0.75, *args, **kwargs):
    constraint_size = max(2, 2+max( (len(graph_connectivity[k]) for k in graph_connectivity) ))

    mdp_2t = nx.DiGraph()

    # init states       # 0 ~ n_s-1             n_s states
    node_label = 0
    for n in range(n_s):
        mdp_2t.add_node(n)
        mdp_2t.nodes[n]['step'] = 0
        mdp_2t.nodes[n]['type'] = 'initial_state'
        mdp_2t.nodes[n]['domain_size'] = 2
        mdp_2t.nodes[n]['parents'] = []   # empty
        mdp_2t.nodes[n]['server_label'] = n
        mdp_2t.nodes[n]['table'] = create_sysadmin_init(init_state=init_state_list[n])

    # decision variables        # n_s ~ 2*n_s -1        n_s actions
    node_label += n_s
    decision_labels = []
    for i, n in enumerate(range(node_label, node_label+n_s)):
        decision_labels.append(n)
        mdp_2t.add_node(n)
        mdp_2t.nodes[n]['step'] = 1
        mdp_2t.nodes[n]['type'] = 'decision'
        mdp_2t.nodes[n]['domain_size'] = 2
        mdp_2t.nodes[n]['parents'] = list(range(n_s))
        mdp_2t.nodes[n]['server_label'] = i
        mdp_2t.nodes[n]['table'] = []

    node_label += n_s           # 2*n_s ~ 3*n_s-1       n_s states
    # transition states
    for i, n in enumerate(range(node_label, node_label+n_s)):
        mdp_2t.add_node(n)
        mdp_2t.nodes[n]['step'] = 2
        mdp_2t.nodes[n]['type'] = 'state'
        mdp_2t.nodes[n]['domain_size'] = 2
        mdp_2t.nodes[n]['action_parents'] = [decision_labels[i]]
        mdp_2t.nodes[n]['state_parents'] = [i]  + graph_connectivity[i]
        mdp_2t.nodes[n]['parents'] = mdp_2t.nodes[n]['action_parents'] + mdp_2t.nodes[n]['state_parents']
        mdp_2t.nodes[n]['server_label'] = i
        mdp_2t.nodes[n]['table'] = create_sysadmin_trans( len(graph_connectivity[i]), reboot_prob )   # don't sort parents


    node_label += n_s
    # utility functions for states      # 3*n_s ~ 4*n_s-1       n_s rewards
    for i, n in enumerate(range(node_label, node_label+n_s)):
        mdp_2t.add_node(n)
        mdp_2t.nodes[n]['step'] = 3
        mdp_2t.nodes[n]['type'] = 'utility'
        mdp_2t.nodes[n]['domain_size'] = None
        mdp_2t.nodes[n]['action_parents'] =[]
        mdp_2t.nodes[n]['state_parents'] = [n - n_s]
        mdp_2t.nodes[n]['parents'] = mdp_2t.nodes[n]['action_parents'] + mdp_2t.nodes[n]['state_parents']
        mdp_2t.nodes[n]['server_label'] = i
        mdp_2t.nodes[n]['table'] = [0.0, 1.0]

    node_label += n_s
    # utility cost for actions          # 4*n_s ~ 5*n_s-1       n_s costs per actions
    for i, n in enumerate(range(node_label, node_label + n_s)):
        mdp_2t.add_node(n)
        mdp_2t.nodes[n]['step'] = 3
        mdp_2t.nodes[n]['type'] = 'utility'
        mdp_2t.nodes[n]['domain_size'] = None
        mdp_2t.nodes[n]['action_parents'] = [n - 3*n_s]
        mdp_2t.nodes[n]['state_parents'] = []
        mdp_2t.nodes[n]['parents'] = mdp_2t.nodes[n]['action_parents'] + mdp_2t.nodes[n]['state_parents']
        mdp_2t.nodes[n]['server_label'] = i
        mdp_2t.nodes[n]['table'] = [0.0, -reboot_penalty]
        # adding +1 for all action variables; n_server * time_horizon negative const to positive
        # mdp_2t.nodes[n]['table'] = [1.0, 1.0-problem_params['reboot_penality']]

    node_label += n_s
    # build constraint tree
    root = constraint_tree(decision_labels, constraint_size)
    l = post_traversal(root)
    tree2dbn = {}
    for node in l:
        if node.action_label:
            tree2dbn[node.node_id] = node.action_label

    for n, node in enumerate( (el for el in l if el.action_label is None), start= node_label):
        mdp_2t.add_node(n)
        tree2dbn[node.node_id] = n      # node.node_id 0 for the root, 1,2,3... for actions and temp follows
        mdp_2t.nodes[n]['step'] = 4
        mdp_2t.nodes[n]['type'] = 'constraint'
        mdp_2t.nodes[n]['domain_size'] = 3       # domain 0 zero selection 1 one selection 2 more than one selection
        mdp_2t.nodes[n]['parents'] = sorted([tree2dbn[k.node_id] for k in node.ch])
        mdp_2t.nodes[n]['server_label'] = None
        mdp_2t.nodes[n]['table'] = create_sysadmin_constraint_rec(mdp_2t.nodes[n]['parents'], decision_labels,
                                                             node.node_id == 0)
    return mdp_2t


def unroll_2t_sysadmin_mdp_e1(mdp_2t:nx.DiGraph, time_horizon, file_name):
    id_diagram = nx.DiGraph()

    initial_states = []
    decision_nodes = []
    utility_nodes = []
    state_nodes = []
    constraint_nodes = []
    for n in mdp_2t:
        node_type = mdp_2t.nodes[n]['type']
        if node_type == 'initial_state':
            initial_states.append(n)
        elif node_type == 'decision':
            decision_nodes.append(n)
        elif node_type == 'state':
            state_nodes.append(n)
        elif node_type == 'utility':
            utility_nodes.append(n)
        elif node_type == 'constraint':
            constraint_nodes.append(n)
    initial_states = sorted(initial_states)
    utility_nodes = sorted(utility_nodes)
    state_nodes = sorted(state_nodes)
    constraint_nodes = sorted(constraint_nodes)
    n_s, n_a, n_u, n_c = len(state_nodes), len(decision_nodes), len(utility_nodes), len(constraint_nodes)

    partial_ordering = []   # specific to this mdp structure
    hidden_nodes = []
    domains = []
    id_node_to_mdp_2t = {}  # influence diagram node to mdp node
    id_node_to_var_id = {}  # influence diagram node to variable id in uai file
    node_label = 0          # node label inside nx graph
    var_id = 0              # var (node) id for uai file
    # init state
    current_state_nodes = []
    for s_node in initial_states:
        id_diagram.add_node(node_label)
        current_state_nodes.append(node_label)
        id_node_to_var_id[node_label] = var_id
        id_node_to_mdp_2t[node_label] = s_node
        domains.append( mdp_2t.nodes[s_node]['domain_size'])
        id_diagram.nodes[node_label]['parents'] = []  # no parents for init state
        node_label += 1
        var_id += 1
    partial_ordering.append(current_state_nodes)

    for th in range(time_horizon):  # from 0 to T      0 is for initial states only
        one_stage_offset = (n_s+n_a+n_u+n_c)*th
        transition_offset = max(0, th*(n_s + n_a) + (th-1)*(n_u + n_c))

        # decision
        current_decision_nodes = []
        for d_node in decision_nodes:
            id_diagram.add_node(node_label)
            id_node_to_var_id[node_label] = var_id
            id_node_to_mdp_2t[node_label] = d_node
            domains.append(mdp_2t.nodes[d_node]['domain_size'])
            current_decision_nodes.append(node_label)

            id_diagram.nodes[node_label]['parents'] = [k + transition_offset for k in mdp_2t.nodes[d_node]['parents']]
            for pa in id_diagram.nodes[node_label]['parents']:
                id_diagram.add_edge(pa, node_label)
            node_label += 1
            var_id += 1
        partial_ordering.append(current_decision_nodes)

        # states
        current_state_nodes = []
        for s_node in state_nodes:
            id_diagram.add_node(node_label)
            id_node_to_var_id[node_label] = var_id
            id_node_to_mdp_2t[node_label] = s_node
            domains.append(mdp_2t.nodes[s_node]['domain_size'])
            current_state_nodes.append(node_label)

            id_diagram.nodes[node_label]['parents'] = [k + one_stage_offset  for k in mdp_2t.nodes[s_node]['action_parents']]
            id_diagram.nodes[node_label]['parents'] += [k + transition_offset for k in mdp_2t.nodes[s_node]['state_parents']]
            for pa in id_diagram.nodes[node_label]['parents']:
                id_diagram.add_edge(pa, node_label)
            node_label += 1
            var_id += 1
        partial_ordering.append(current_state_nodes)

        # utility
        for u_node in utility_nodes:
            id_diagram.add_node(node_label)
            id_node_to_mdp_2t[node_label] = u_node

            id_diagram.nodes[node_label]['parents'] = [k + one_stage_offset for k in mdp_2t.nodes[u_node]['parents']]
            for pa in id_diagram.nodes[node_label]['parents']:
                id_diagram.add_edge(pa, node_label)
            node_label += 1

        # constraints    acting as hidden variables
        for c_node in constraint_nodes:
            id_diagram.add_node(node_label)
            id_node_to_var_id[node_label] = var_id
            id_node_to_mdp_2t[node_label] = c_node
            domains.append(mdp_2t.nodes[c_node]['domain_size'])
            hidden_nodes.append(node_label)

            id_diagram.nodes[node_label]['parents'] = [k + one_stage_offset for k in mdp_2t.nodes[c_node]['parents']]
            for pa in id_diagram.nodes[node_label]['parents']:
                id_diagram.add_edge(pa, node_label)
            node_label += 1
            var_id += 1

    partial_ordering.append(hidden_nodes)

    write_id_for_fhmdp(file_name, time_horizon, domains, partial_ordering,
                       n_s, n_a, n_u, n_c,
                       mdp_2t, id_diagram, id_node_to_mdp_2t, id_node_to_var_id)
    return id_diagram


def write_id_for_fhmdp(file_name, time_horizon, domains, partial_ordering,
                       num_state_nodes, num_decision_nodes, num_util_nodes, num_constraint_nodes,
                       mdp_2t:nx.DiGraph, id_diagram:nx.DiGraph,
                       id_node_to_mdp, id_node_to_var_id):
    f = open(file_name + '.uai', 'w')
    f.write('ID\n')
    # vars
    n_vars = sum([len(bk) for bk in partial_ordering] )
    f.write('{}\n'.format(n_vars))
    for d in domains:
        f.write('{} '.format(d))
    f.write('\n')
    n_factors = num_state_nodes + (num_state_nodes + num_util_nodes + num_constraint_nodes) * time_horizon
    f.write('{}\n'.format(n_factors))
    for n in sorted(iter(id_diagram)):
        if mdp_2t.nodes[id_node_to_mdp[n]]['type'] in ['initial_state', 'state', 'constraint']:
            scope = id_diagram.nodes[n]['parents'] + [n]
        elif mdp_2t.nodes[id_node_to_mdp[n]]['type'] == 'utility':
            scope = id_diagram.nodes[n]['parents']
        else:
            continue
        f.write('{} '.format(len(scope)))
        for s in scope:
            f.write('{} '.format(id_node_to_var_id[s]))
        f.write('\n')
    f.write('\n')
    # tables
    for n in sorted(iter(id_diagram)):
        if mdp_2t.nodes[id_node_to_mdp[n]]['type'] != 'decision':
            f.write('{}\n'.format(len(mdp_2t.nodes[id_node_to_mdp[n]]['table'])))
            for t in mdp_2t.nodes[id_node_to_mdp[n]]['table']:
                f.write('{}\n'.format(float(t)))
            f.write('\n')
    f.close()

    f = open(file_name + '.id', 'w')
    f.write('{}\n'.format(n_vars))
    # var types
    for n in sorted(iter(id_diagram)):
        if mdp_2t.nodes[id_node_to_mdp[n]]['type'] in ['initial_state', 'state', 'constraint']:
            f.write('C ')
        elif mdp_2t.nodes[id_node_to_mdp[n]]['type'] == 'decision':
            f.write('D ')
    f.write('\n')
    # factor types
    f.write('{}\n'.format(n_factors))
    for n in sorted(iter(id_diagram)):
        if mdp_2t.nodes[id_node_to_mdp[n]]['type'] in ['initial_state', 'state', 'constraint']:
            f.write('P ')
        elif mdp_2t.nodes[id_node_to_mdp[n]]['type'] == 'utility':
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


def make_2t_sysadmin_mdp_e2(n_s, graph_connectivity, init_state_list: List[bool], reboot_prob=0.1, reboot_penalty=0.75,
                            constraint_violation_cost=100):
    # if not is_table_constraint:
    #     constraint_size = n_s
    # else:
    constraint_size = max(2, 2 + max((len(graph_connectivity[k]) for k in graph_connectivity)))

    mdp_2t = nx.DiGraph()

    # init states       # 0 ~ n_s-1             n_s states
    node_label = 0
    for n in range(n_s):
        mdp_2t.add_node(n)
        mdp_2t.nodes[n]['step'] = 0
        mdp_2t.nodes[n]['type'] = 'initial_state'
        mdp_2t.nodes[n]['domain_size'] = 2
        mdp_2t.nodes[n]['parents'] = []  # empty
        mdp_2t.nodes[n]['server_label'] = n
        mdp_2t.nodes[n]['table'] = create_sysadmin_init(init_state=init_state_list[n])

    # decision variables        # n_s ~ 2*n_s -1        n_s actions
    node_label += n_s
    decision_labels = []
    for i, n in enumerate(range(node_label, node_label + n_s)):
        decision_labels.append(n)
        mdp_2t.add_node(n)
        mdp_2t.nodes[n]['step'] = 1
        mdp_2t.nodes[n]['type'] = 'decision'
        mdp_2t.nodes[n]['domain_size'] = 2
        mdp_2t.nodes[n]['parents'] = list(range(n_s))
        mdp_2t.nodes[n]['server_label'] = i
        mdp_2t.nodes[n]['table'] = []

    node_label += n_s  # 2*n_s ~ 3*n_s-1       n_s states
    # transition states
    for i, n in enumerate(range(node_label, node_label + n_s)):
        mdp_2t.add_node(n)
        mdp_2t.nodes[n]['step'] = 2
        mdp_2t.nodes[n]['type'] = 'state'
        mdp_2t.nodes[n]['domain_size'] = 2
        mdp_2t.nodes[n]['action_parents'] = [decision_labels[i]]
        mdp_2t.nodes[n]['state_parents'] = [i] + graph_connectivity[i]
        mdp_2t.nodes[n]['parents'] = mdp_2t.nodes[n]['action_parents'] + mdp_2t.nodes[n]['state_parents']
        mdp_2t.nodes[n]['server_label'] = i
        mdp_2t.nodes[n]['table'] = create_sysadmin_trans(len(graph_connectivity[i]), reboot_prob)  # don't sort parents

    node_label += n_s
    # build constraint tree
    root = constraint_tree(decision_labels, constraint_size)
    l = post_traversal(root)
    tree2dbn = {}
    for node in l:
        if node.action_label:
            tree2dbn[node.node_id] = node.action_label

    for n, node in enumerate((el for el in l if el.action_label is None), start=node_label):
        mdp_2t.add_node(n)
        tree2dbn[node.node_id] = n  # node.node_id 0 for the root, 1,2,3... for actions and temp follows
        mdp_2t.nodes[n]['step'] = 3
        mdp_2t.nodes[n]['type'] = 'constraint'
        mdp_2t.nodes[n]['domain_size'] = 3  # domain 0 zero selection 1 one selection 2 more than one selection
        mdp_2t.nodes[n]['parents'] = sorted([tree2dbn[k.node_id] for k in node.ch])
        mdp_2t.nodes[n]['server_label'] = None
        mdp_2t.nodes[n]['table'] = create_sysadmin_constraint_rec2(mdp_2t.nodes[n]['parents'], decision_labels,
                                                                  node.node_id == 0)

    constraint_root_node = tree2dbn[root.node_id]  # this is now parent to all utility
    n_latent = len(l) - n_s
    # domain value is 0, 1, 2

    node_label += n_latent
    # utility functions for states      # 3*n_s ~ 4*n_s-1       n_s rewards
    for i, n in enumerate(range(node_label, node_label + n_s)):
        mdp_2t.add_node(n)
        mdp_2t.nodes[n]['step'] = 4
        mdp_2t.nodes[n]['type'] = 'utility'
        mdp_2t.nodes[n]['domain_size'] = None
        mdp_2t.nodes[n]['action_parents'] = []
        mdp_2t.nodes[n]['state_parents'] = [n - n_s - n_latent]
        mdp_2t.nodes[n]['parents'] = [constraint_root_node] + mdp_2t.nodes[n]['state_parents']
        mdp_2t.nodes[n]['server_label'] = i
        mdp_2t.nodes[n]['table'] = [0.0, 1.0] + [0.0, 1.0] + [-constraint_violation_cost] * 2

    node_label += n_s
    # utility cost for actions          # 4*n_s ~ 5*n_s-1       n_s costs per actions
    for i, n in enumerate(range(node_label, node_label + n_s)):
        mdp_2t.add_node(n)
        mdp_2t.nodes[n]['step'] = 4
        mdp_2t.nodes[n]['type'] = 'utility'
        mdp_2t.nodes[n]['domain_size'] = None
        mdp_2t.nodes[n]['action_parents'] = [n - 3 * n_s - n_latent]
        mdp_2t.nodes[n]['state_parents'] = []
        mdp_2t.nodes[n]['parents'] = [constraint_root_node] + mdp_2t.nodes[n]['action_parents']
        mdp_2t.nodes[n]['server_label'] = i
        # mdp_2t.nodes[n]['table'] = [-constraint_violation_cost] * 2 + [0.0, -reboot_penalty] + [-constraint_violation_cost] * 2
        mdp_2t.nodes[n]['table'] = [0.0, -constraint_violation_cost, 0.0, -reboot_penalty,
                                    -constraint_violation_cost, -constraint_violation_cost]
        # adding +1 for all action variables; n_server * time_horizon negative const to positive
        # mdp_2t.nodes[n]['table'] = [1.0, 1.0-problem_params['reboot_penality']]

    return mdp_2t


def unroll_2t_sysadmin_mdp_e2(mdp_2t: nx.DiGraph, time_horizon, file_name):
    id_diagram = nx.DiGraph()

    initial_states = []
    decision_nodes = []
    utility_nodes = []
    state_nodes = []
    constraint_nodes = []
    for n in mdp_2t:
        node_type = mdp_2t.nodes[n]['type']
        if node_type == 'initial_state':
            initial_states.append(n)
        elif node_type == 'decision':
            decision_nodes.append(n)
        elif node_type == 'state':
            state_nodes.append(n)
        elif node_type == 'utility':
            utility_nodes.append(n)
        elif node_type == 'constraint':
            constraint_nodes.append(n)
    initial_states = sorted(initial_states)
    utility_nodes = sorted(utility_nodes)
    state_nodes = sorted(state_nodes)
    constraint_nodes = sorted(constraint_nodes)
    n_s, n_a, n_u, n_c = len(state_nodes), len(decision_nodes), len(utility_nodes), len(constraint_nodes)

    partial_ordering = []  # specific to this mdp structure
    hidden_nodes = []
    domains = []
    id_node_to_mdp_2t = {}  # influence diagram node to mdp node
    id_node_to_var_id = {}  # influence diagram node to variable id in uai file
    node_label = 0  # node label inside nx graph
    var_id = 0  # var (node) id for uai file
    # init state
    current_state_nodes = []
    for s_node in initial_states:
        id_diagram.add_node(node_label)
        current_state_nodes.append(node_label)
        id_node_to_var_id[node_label] = var_id
        id_node_to_mdp_2t[node_label] = s_node
        domains.append(mdp_2t.nodes[s_node]['domain_size'])
        id_diagram.nodes[node_label]['parents'] = []  # no parents for init state
        node_label += 1
        var_id += 1
    partial_ordering.append(current_state_nodes)

    for th in range(time_horizon):  # from 0 to T      0 is for initial states only
        one_stage_offset = (n_s + n_a + n_u + n_c) * th
        transition_offset = max(0, th * (n_s + n_a) + (th - 1) * (n_u + n_c))

        # decision
        current_decision_nodes = []
        for d_node in decision_nodes:
            id_diagram.add_node(node_label)
            id_node_to_var_id[node_label] = var_id
            id_node_to_mdp_2t[node_label] = d_node
            domains.append(mdp_2t.nodes[d_node]['domain_size'])
            current_decision_nodes.append(node_label)

            id_diagram.nodes[node_label]['parents'] = [k + transition_offset for k in mdp_2t.nodes[d_node]['parents']]
            for pa in id_diagram.nodes[node_label]['parents']:
                id_diagram.add_edge(pa, node_label)
            node_label += 1
            var_id += 1
        partial_ordering.append(current_decision_nodes)

        # states
        current_state_nodes = []
        for s_node in state_nodes:
            id_diagram.add_node(node_label)
            id_node_to_var_id[node_label] = var_id
            id_node_to_mdp_2t[node_label] = s_node
            domains.append(mdp_2t.nodes[s_node]['domain_size'])
            current_state_nodes.append(node_label)

            id_diagram.nodes[node_label]['parents'] = [k + one_stage_offset for k in
                                                       mdp_2t.nodes[s_node]['action_parents']]
            id_diagram.nodes[node_label]['parents'] += [k + transition_offset for k in
                                                        mdp_2t.nodes[s_node]['state_parents']]
            for pa in id_diagram.nodes[node_label]['parents']:
                id_diagram.add_edge(pa, node_label)
            node_label += 1
            var_id += 1
        partial_ordering.append(current_state_nodes)

        # constraints    acting as hidden variables
        for c_node in constraint_nodes:
            id_diagram.add_node(node_label)
            id_node_to_var_id[node_label] = var_id
            id_node_to_mdp_2t[node_label] = c_node
            domains.append(mdp_2t.nodes[c_node]['domain_size'])
            hidden_nodes.append(node_label)

            id_diagram.nodes[node_label]['parents'] = [k + one_stage_offset for k in mdp_2t.nodes[c_node]['parents']]
            for pa in id_diagram.nodes[node_label]['parents']:
                id_diagram.add_edge(pa, node_label)
            node_label += 1
            var_id += 1

        # utility
        for u_node in utility_nodes:
            id_diagram.add_node(node_label)
            id_node_to_mdp_2t[node_label] = u_node

            id_diagram.nodes[node_label]['parents'] = [k + one_stage_offset for k in mdp_2t.nodes[u_node]['parents']]
            for pa in id_diagram.nodes[node_label]['parents']:
                id_diagram.add_edge(pa, node_label)
            node_label += 1

    partial_ordering.append(hidden_nodes)

    write_id_for_fhmdp(file_name, time_horizon, domains, partial_ordering,
                       n_s, n_a, n_u, n_c,
                       mdp_2t, id_diagram, id_node_to_mdp_2t, id_node_to_var_id)
    return id_diagram

if __name__ == "__main__":
    config_num = "a"
    n_servers=2
    reboot_prob = 0.1
    reboot_penalty = 0.75
    g = {
        0: [1],
        1: [],
    }

    constraint_violation_cost = 100

    for time_horizon in range(3, 4):
        problem_path = os.path.join(BENCHMARK_DIR, "sysadmin_mdp")
        # problem_name = "_".join(["sysadmin_inst_mdp", str(config_num), "s="+str(n_servers),"t="+str(time_horizon)])
        problem_name = "_".join(["sysadmin_mdp", str(config_num), "s="+str(n_servers),"t="+str(time_horizon)])
        file_name = os.path.join(problem_path, problem_name)

        pomdp_2t = make_2t_sysadmin_mdp_e2(n_servers, g, [True] * n_servers, reboot_prob, reboot_penalty, constraint_violation_cost)
        unroll_2t_sysadmin_mdp_e2(pomdp_2t, time_horizon=time_horizon, file_name=file_name)
