"""

read or write uai files in pure python classes

*.uai: functions
*.id: identity for var/func
*.mi: identify for var
*.map: map variables
*.pvo: partial ordering
*.evid: evidence or conditioning
*.vo: total ordering
*.pt: pseudo tree
* ID/LIMID: uai, id, pvo
* MMAP: uai, map
* MPE: uai
* SUM: uai
* MI: uai, mi
"""
from typing import Text, List
from gmid2.global_constants import *


class FileInfo:
    def __init__(self):
        self.uai_file = ""
        self.net_type = ""
        self.nvar, self.nfunc, self.nchance, self.ndec, self.nprob, self.nutil = 0, 0, 0, 0, 0, 0
        self.var_types, self.domains = [], []
        self.chance_vars, self.decision_vars = [], []
        self.prob_funcs, self.util_funcs = [], []
        self.tables, self.scopes, self.factor_types = [], [], []
        self.blocks, self.block_types, self.nblock = [], [], 0

    def show_members(self):
        for k, v in vars(self).items():
            print("{}:{}".format(k,v))


def read_limid(file_name: Text, skip_table: bool=False) -> FileInfo:
    """
    read necessary files (uai for vars/functions, id for identity, pvo for partil order) for loading limid
    use the return to create graphical model object that actually creates numpy based Factors
    """
    file_info = FileInfo()
    assert not file_name.endswith(".uai"), "remove extension as this read *.uai, *.id, *.pvo in sequence"
    read_uai(file_name + ".uai", file_info, skip_table)
    read_id(file_name + ".id", file_info)
    read_pvo(file_name + ".pvo", file_info)
    return file_info


def read_mmap(file_name:Text, skip_table: bool=False)->FileInfo:
    """
    read necessary files (uai for vars/functions, map for max variables, pvo is implicit)
    """
    file_info = FileInfo()
    read_uai(file_name + ".uai", file_info, skip_table)
    read_map(file_name + ".map", file_info)
    return file_info


def read_sum(file_name:Text, skip_table: bool=False)->FileInfo:
    file_info = FileInfo()
    read_uai(file_name + ".uai", file_info, skip_table)

    file_info.ndec = 0
    file_info.decision_vars = []
    file_info.chance_vars = list(range(file_info.nvar))
    file_info.nchance = file_info.nvar
    file_info.var_types = [TYPE_CHANCE_VAR] * file_info.nchance
    file_info.factor_types = [TYPE_PROB_FUNC] * file_info.nfunc
    file_info.prob_funcs = list(range(file_info.nfunc))
    file_info.nprob = file_info.nfunc
    file_info.util_funcs = []
    file_info.nutil = 0
    file_info.nblock = 1
    file_info.block_types = [TYPE_CHANCE_BLOCK]
    file_info.blocks = list(file_info.chance_vars)
    return file_info


def read_mpe(file_name: Text, skip_table: bool = False) -> FileInfo:
    file_info = FileInfo()
    read_uai(file_name + ".uai", file_info, skip_table)
    file_info.nchance = 0
    file_info.chance_vars = []
    file_info.decision_vars = list(range(file_info.nvar))
    file_info.ndec = file_info.nvar
    file_info.var_types = [TYPE_DECISION_VAR] * file_info.nvar
    file_info.factor_types = [TYPE_PROB_FUNC] * file_info.nfunc
    file_info.prob_funcs = list(range(file_info.nfunc))
    file_info.nprob = file_info.nfunc
    file_info.util_funcs = []
    file_info.nutil = 0
    file_info.nblock = 1
    file_info.block_types = [TYPE_CHANCE_BLOCK]
    file_info.blocks = list(file_info.chance_vars)
    return file_info


def read_mixed(file_name: Text, skip_table: bool = False) -> FileInfo:
    file_info = FileInfo()
    assert not file_name.endswith(".uai"), "remove extension as this read *.uai, *.mi, *.pvo in sequence"
    read_uai(file_name + ".uai", file_info, skip_table)
    read_mi(file_name + ".mi", file_info)
    read_pvo(file_name + ".pvo", file_info)
    return file_info


def read_uai(file_name: Text, file_info: FileInfo=None, skip_table: bool=False) -> FileInfo:
    if file_info is None:
        file_info = FileInfo()
    with open(file_name) as file_iter:
        token = get_token(file_iter)
        file_info.uai_file = file_name.split('/')[-1]
        file_info.net_type = next(token).upper()                            # expect to see ID, LIMID but BN, MN also works
        file_info.nvar = int(next(token))
        file_info.domains = [int(next(token)) for _ in range(file_info.nvar)]       # domain size of variables
        file_info.nfunc = int(next(token))
        for _ in range(file_info.nfunc):
            scope_size = int(next(token))                           # num vars appear in each function
            scope = [int(next(token)) for _ in range(scope_size)]   # list of var_ids from 0 to nvar-1
            file_info.scopes.append(scope)
        if not skip_table:
            for _ in range(file_info.nfunc):
                nrow = int(next(token))
                table = [float(next(token)) + ZERO for _ in range(nrow)]
                file_info.tables.append(table)
        else:
            for _ in range(file_info.nfunc):
                file_info.tables.append(None)

    return file_info


def read_id(file_name: Text, file_info: FileInfo) -> FileInfo:
    """
    Read identity of variables and functions
    """
    with open(file_name) as file_iter:
        token = get_token(file_iter)
        nvar = int(next(token))
        assert nvar == file_info.nvar, "number of variables don't match"
        file_info.var_types = [next(token).upper() for _ in range(nvar)]
        nfunc = int(next(token))
        assert nfunc == file_info.nfunc, "number of factors don't match"
        file_info.factor_types = [next(token).upper() for _ in range(nfunc)]

        file_info.chance_vars = [i for i, el in enumerate(file_info.var_types) if el == TYPE_CHANCE_VAR]
        file_info.nchance = len(file_info.chance_vars)
        file_info.decision_vars = [i for i, el in enumerate(file_info.var_types) if el == TYPE_DECISION_VAR]
        file_info.ndec = len(file_info.decision_vars)

        file_info.util_funcs = [i for i, el in enumerate(file_info.factor_types) if el == TYPE_UTIL_FUNC]
        file_info.nutil = len(file_info.util_funcs)
        file_info.prob_funcs = [i for i, el in enumerate(file_info.factor_types) if el in [TYPE_PROB_FUNC, TYPE_CHANCE_VAR]]
        file_info.nprob = len(file_info.prob_funcs)
        assert file_info.nprob == file_info.nchance, "number of probability functions don't match"
    assert file_info.nvar == file_info.nchance + file_info.ndec, "number of variables don't match"
    assert file_info.nfunc == file_info.nprob + file_info.nutil, "number of functions don't match"
    return file_info


def read_map(file_name: Text, file_info: FileInfo) -> FileInfo:
    with open(file_name) as file_iter:
        token = get_token(file_iter)
        file_info.ndec = int(next(token))
        file_info.decision_vars = [int(next(token)) for _ in range(file_info.ndec)]
        dec_vars_set = set(file_info.decision_vars)
        file_info.chance_vars = [i for i in range(file_info.nvar) if i not in dec_vars_set]
        chance_var_set = set(file_info.chance_vars)
        file_info.nchance = len(file_info.chance_vars)

        file_info.var_types = []
        for i in range(file_info.nvar):
            if i in chance_var_set:
                file_info.var_types.append(TYPE_CHANCE_VAR)
            else:
                file_info.var_types.append(TYPE_DECISION_VAR)

        file_info.util_funcs = []
        file_info.nutil = 0
        file_info.prob_funcs = [i for i in range(file_info.nfunc)]      # mmap all functions are considered prob.
        file_info.nprob = len(file_info.prob_funcs)
    assert file_info.nvar == file_info.nchance + file_info.ndec, "number of variables don't match"

    file_info.factor_types = [TYPE_PROB_FUNC] * file_info.nfunc
    file_info.nblock = 2
    file_info.block_types = [TYPE_CHANCE_BLOCK, TYPE_DEC_BLOCK]
    file_info.blocks = [ list(iter(file_info.chance_vars)), list(iter(file_info.decision_vars))  ]
    return file_info


def read_mi(file_name: Text, file_info: FileInfo) -> FileInfo:
    with open(file_name) as file_iter:
        token = get_token(file_iter)
        nvar = int(next(token))
        assert nvar == file_info.nvar, "number of variables don't match"
        file_info.var_types = [next(token).upper() for _ in range(nvar)]
        file_info.factor_types = [TYPE_PROB_FUNC] * file_info.nfunc

        file_info.chance_vars = [i for i, el in enumerate(file_info.var_types) if el == TYPE_CHANCE_VAR]
        file_info.nchance = len(file_info.chance_vars)
        file_info.decision_vars = [i for i, el in enumerate(file_info.var_types) if el == TYPE_DECISION_VAR]
        file_info.ndec = len(file_info.decision_vars)

        file_info.util_funcs = []
        file_info.nutil = 0
        file_info.prob_funcs = list(range(file_info.nfunc))
        file_info.nprob = file_info.nfunc
    return file_info


def read_pvo(file_name: Text, file_info: FileInfo) -> FileInfo:
    """
    Read block structure of limids, alternating chance and decision variables

    :param file_name:
    :param file_info:
    :return:
    """
    with open(file_name) as file_iter:
        block = get_block(file_iter)
        nvar = int(next(block))
        assert nvar == file_info.nvar, "number of variables don't match"
        file_info.nblock = int(next(block))
        for _ in range(file_info.nblock):
            file_info.blocks.append( [int(el) for el in next(block).split()] )
    for each_block in file_info.blocks:
        if all(el in file_info.decision_vars for el in each_block):  # all variables are decision variables
            file_info.block_types.append(TYPE_DEC_BLOCK)
        else:
            file_info.block_types.append(TYPE_CHANCE_BLOCK)

    assert nvar == sum( [len(b) for b in file_info.blocks] ), "number of variables don't match"
    return file_info


def read_vo(file_name: Text, file_info: FileInfo)->FileInfo:
    with open(file_name, 'r') as file_iter:
        token = get_token(file_iter)
        nvar = int(next(token))
        vo = [int(next(token)) for _ in range(nvar)]
        file_info.vo = vo
    return file_info


def read_evid(file_name: Text, file_info: FileInfo):
    with open(file_name, 'r') as file_iter:
        token = get_token(file_iter)
        nevid = int(next(token))
        evid = [(int(next(token)), int(next(token))) for _ in range(nevid)]
        file_info.evid = evid
    return file_info


def read_pt(file_name: Text, file_info: FileInfo):
    with open(file_name, 'r') as file_iter:
        token = get_token(file_iter)
        nvar = int(next(token))
        pt = [int(next(token)) for _ in range(nvar)]
        file_info.pt = pt
    return file_info


def read_svo(file_name: Text, file_info: FileInfo)->FileInfo:
    with open(file_name, 'r') as file_iter:
        token = get_token(file_iter)
        nsubmodel = int(next(token))
        nvars, widths = [], []
        for _ in range(nsubmodel):
            nvars.append(int(next(token)))      # number of elim variables in submodel
        for _ in range(nsubmodel):
            widths.append(int(next(token)))    # induced width per submodel
        svo = []
        for _ in range(nsubmodel):
            nvar = int(next(token))
            vo = [int(next(token)) for _ in range(nvar)]
            svo.append(vo)
        file_info.svo = svo
        file_info.widths = widths
    return file_info


def write_vo(file_name: Text, elim_order:List[int], induced_width:int=None)->None:
    with open(file_name, 'w') as file_iter:
        file_iter.write("# iw={}\n".format(induced_width))
        file_iter.write(("{}\n".format(len(elim_order))))
        for vid in elim_order:
            file_iter.write("{}\n".format(vid))


def write_pvo_from_partial_elim_order(file_name: Text, partial_variable_ordering: List[List[int]])->None:
    with open(file_name, 'w') as pvo_file:
        # partial variable ordering defines blocks of variables
        pvo_list = [el for el in partial_variable_ordering if len(el) > 0]      # exclude zero length sub-list
        num_blocks = len(pvo_list)
        num_var = max(max(el) for el in pvo_list) + 1                           # total var = largest var id + 1
        pvo_file.write("{};\n".format(num_var))
        pvo_file.write("{};\n".format(num_blocks))
        for block in pvo_list:
            pvo_file.write("{};\n".format(" ".join((str(v) for v in block))))


def write_uai(file_name, file_info: FileInfo, file_type:Text)->None:
    with open(file_name, 'w') as uai_file:
        uai_file.write("{}\n".format(file_type))        # ID, BAYES, MARKOV, LIMID
        uai_file.write("{}\n".format(file_info.nvar))
        uai_file.write("{}\n".format(" ".join(str(el) for el in file_info.domains)))
        uai_file.write("{}\n".format(file_info.nfunc))
        for each_scope in file_info.scopes:
            uai_file.write("{}\n".format(' '.join([str(len(each_scope))]+[str(el) for el in each_scope])))
        uai_file.write("\n")
        for each_factor in file_info.tables:
            uai_file.write(("{}\n".format(len(each_factor))))
            for el in each_factor:
                uai_file.write("{}\n".format(el))
            uai_file.write("\n")


def write_id(file_name: Text, var_types: List[Text], func_types: List[Text])->None:
    with open(file_name, 'w') as id_file:
        id_file.write("{}\n".format(len(var_types)))
        id_file.write("{}\n".format(" ".join((el.upper() for el in var_types))))     # C, D
        id_file.write("{}\n".format(len(func_types)))
        id_file.write("{}\n".format(" ".join((el.upper() for el in func_types))))     # P, U


def write_mi(file_name: Text, var_types: List[Text])->None:
    with open(file_name, 'w') as mi_file:
        mi_file.write("{}\n".format(len(var_types)))
        mi_file.write("{}\n".format(" ".join((el.upper() for el in var_types))))


def write_map_from_types(file_name: Text, var_types: List[Text])->None:
    with open(file_name, 'w') as map_file:
        dec_vars = [str(el) for el in range(len(var_types)) if var_types[el] == "D"]
        map_file.write("{}\n".format(len(dec_vars)))
        map_file.write("{}\n".format("\n".join(dec_vars)))


def write_svo(file_name:Text, svo:List[List[int]], widths:List[int])->None:
    with open(file_name, 'w') as file_iter:
        file_iter.write("{}\n".format(len(svo)))
        file_iter.write("{}\n".format(" ".join( str(len(svo[i])) for i in range(len(svo)))))
        file_iter.write("{}\n".format(" ".join( str(i) for i in widths )))
        for vo in svo:
            file_iter.write("{} {}\n".format(len(vo), " ".join(str(i) for i in vo)))

def write_evid():
    raise NotImplementedError


def write_pt():
    raise NotImplementedError


def get_line(file_iter):
    for each_line in file_iter:
        line = each_line.strip()
        if line.startswith("#") or line.startswith("//"):       # skip comment lines
            continue
        if line:
            yield line


def get_token(file_iter):
    for each_line in get_line(file_iter):
        for each_token in each_line.split():
            if each_token:
                yield each_token


def get_block(file_iter):
    for each_block in file_iter.read().split(';'):
        each_block = each_block.strip()
        if each_block.startswith("#") or each_block.startswith("//"):
            continue
        if each_block:
            yield each_block


def standardize_util(file_name: Text, weight:int=1):
    file_info = read_limid(file_name)
    new_file_name = file_name + "_std_" + str(weight)
    values = []
    for ind, t in enumerate(file_info.factor_types):
        if t == TYPE_UTIL_FUNC:
            values.extend(file_info.tables[ind])
            file_info.tables[ind] = np.array([file_info.tables[ind]]).flatten()
    values = np.array(values)
    mu = np.mean(values)
    sigma = np.std(values)

    for ind, t in enumerate(file_info.factor_types):
        if t == TYPE_UTIL_FUNC:
            file_info.tables[ind] = (file_info.tables[ind] - mu) / (sigma * weight)
            file_info.tables[ind] = list(file_info.tables[ind])

    write_uai(new_file_name + ".uai", file_info, file_info.net_type)
    write_id(new_file_name + ".id", file_info.var_types, file_info.factor_types)
    write_pvo_from_partial_elim_order(new_file_name + ".pvo", file_info.blocks)


def rescale_util(file_name: Text, s=1.0):
    file_info = read_limid(file_name)
    new_file_name = file_name + "_norm_" + str(s)
    values = []
    for ind, t in enumerate(file_info.factor_types):
        if t == TYPE_UTIL_FUNC:
            values.extend(file_info.tables[ind])
            file_info.tables[ind] = np.array([file_info.tables[ind]]).flatten()
    values = np.array(values)
    min_v = np.min(values)
    max_v = np.max(values)

    for ind, t in enumerate(file_info.factor_types):
        if t == TYPE_UTIL_FUNC:
            file_info.tables[ind] = (file_info.tables[ind] - min_v + EPSILON) / (max_v - min_v)
            file_info.tables[ind] = file_info.tables[ind]/s
            file_info.tables[ind] = list(file_info.tables[ind])

    write_uai(new_file_name + ".uai", file_info, file_info.net_type)
    write_id(new_file_name + ".id", file_info.var_types, file_info.factor_types)
    write_pvo_from_partial_elim_order(new_file_name + ".pvo", file_info.blocks)


def rescale_center_util(file_name: Text, s=1.0):
    file_info = read_limid(file_name)
    new_file_name = file_name + "_norm_center_" + str(int(s))
    values = []
    for ind, t in enumerate(file_info.factor_types):
        if t == TYPE_UTIL_FUNC:
            values.extend(file_info.tables[ind])
            file_info.tables[ind] = np.array([file_info.tables[ind]]).flatten()
    values = np.array(values)
    min_v = np.min(values)
    max_v = np.max(values)

    for ind, t in enumerate(file_info.factor_types):
        if t == TYPE_UTIL_FUNC:
            file_info.tables[ind] = (file_info.tables[ind] - min_v) / (max_v - min_v)
            file_info.tables[ind] = file_info.tables[ind] - 0.5     # center around zero
            file_info.tables[ind] = file_info.tables[ind]/s         # scale down
            file_info.tables[ind] = list(file_info.tables[ind])

    write_uai(new_file_name + ".uai", file_info, file_info.net_type)
    write_id(new_file_name + ".id", file_info.var_types, file_info.factor_types)
    write_pvo_from_partial_elim_order(new_file_name + ".pvo", file_info.blocks)


def rescale_util_round(file_name: Text, s=10):
    file_info = read_limid(file_name)
    new_file_name = file_name + "_round_norm_" + str(s)
    values = []
    for ind, t in enumerate(file_info.factor_types):
        if t == TYPE_UTIL_FUNC:
            values.extend(file_info.tables[ind])
            file_info.tables[ind] = np.array([file_info.tables[ind]]).flatten()
    values = np.array(values)
    min_v = np.min(values)
    max_v = np.max(values)

    for ind, t in enumerate(file_info.factor_types):
        if t == TYPE_UTIL_FUNC:
            file_info.tables[ind] = (file_info.tables[ind] - min_v) / (max_v - min_v)       # btw 0 and 1
            file_info.tables[ind] = file_info.tables[ind] * s
            file_info.tables[ind] = np.round(file_info.tables[ind])
            file_info.tables[ind][file_info.tables[ind]==0.0] = EPSILON
            file_info.tables[ind] = list(file_info.tables[ind])

    write_uai(new_file_name + ".uai", file_info, file_info.net_type)
    write_id(new_file_name + ".id", file_info.var_types, file_info.factor_types)
    write_pvo_from_partial_elim_order(new_file_name + ".pvo", file_info.blocks)


def change_file_type(file_name: Text, new_file_name:Text, file_type:Text):
    file_info = read_limid(file_name)

    write_uai(new_file_name + ".uai", file_info, file_type)
    write_id(new_file_name + ".id", file_info.var_types, file_info.factor_types)
    write_pvo_from_partial_elim_order(new_file_name + ".pvo", file_info.blocks)


