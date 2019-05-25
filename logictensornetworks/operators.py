import tensorflow as tf

from logictensornetworks import cross_args, F_And, F_Or, cross_2args, F_Implies, F_Not, F_Equiv, F_Forall, F_Exists

DEFAULT_TNORM = "luk"
DEFAULT_UNIVERSAL_AGG = "hmean"
DEFAULT_EXISTENTIAL_AGGREGATOR = "max"

def set_tnorm(tnorm):
    assert tnorm in ['min','luk','prod','mean','']
    global F_And,F_Or,F_Implies,F_Not,F_Equiv,F_Forall
    if tnorm == "min":
        def F_And(wffs):
            return tf.reduce_min(wffs,axis=-1,keepdims=True)

        def F_Or(wffs):
            return tf.reduce_max(wffs,axis=-1,keepdims=True)

        def F_Implies(wff1, wff2):
            return tf.maximum(tf.to_float(tf.less_equal(wff1, wff2)), wff2)

        def F_Not(wff):
            return 1 - wff

        def F_Equiv(wff1,wff2):
            return tf.maximum(tf.to_float(tf.equal(wff1,wff2)),tf.minimum(wff1,wff2))

    if tnorm == "prod":
        def F_And(wffs):
            return tf.reduce_prod(wffs,axis=-1,keepdims=True)

        def F_Or(wffs):
            return 1-tf.reduce_prod(1-wffs,axis=-1,keepdims=True)

        def F_Implies(wff1, wff2):
            le_wff1_wff2 = tf.to_float(tf.less_equal(wff1,wff2))
            gt_wff1_wff2 = tf.to_float(tf.greater(wff1,wff2))
            return tf.cond(tf.equal(wff1[0],0),lambda:le_wff1_wff2 + gt_wff1_wff2*wff2/wff1,lambda:tf.constant([1.0]))


        def F_Not(wff):
            # according to standard goedel logic is
            # return tf.to_float(tf.equal(wff,1))
            return 1-wff

        def F_Equiv(wff1,wff2):
            return tf.minimum(wff1/wff2,wff2/wff1)

    if tnorm == "mean":
        def F_And(wffs):
            return tf.reduce_mean(wffs,axis=-1,keepdims=True)

        def F_Or(wffs):
            return tf.reduce_max(wffs,axis=-1,keepdims=True)

        def F_Implies(wff1, wff2):
            return tf.clip_by_value(2*wff2-wff1,0,1)

        def F_Not(wff):
            return 1 - wff

        def F_Equiv(wff1,wff2):
            return 1 - tf.abs(wff1-wff2)

    if tnorm == "luk":
        def F_And(wffs):
            return tf.maximum(0.0,tf.reduce_sum(wffs,axis=-1,keepdims=True)+1-tf.to_float(tf.shape(wffs)[-1]))

        def F_Or(wffs):
            return tf.minimum(tf.reduce_sum(wffs,axis=-1,keepdims=True),1.0,)

        def F_Implies(wff1, wff2):
            return tf.minimum(1.,1 - wff1 + wff2)

        def F_Not(wff):
            return 1 - wff

        def F_Equiv(wff1,wff2):
            return 1 - tf.abs(wff1-wff2)


def set_universal_aggreg(aggreg):
    assert aggreg in ['hmean','min','mean']
    global F_Forall
    if aggreg == "hmean":
        def F_Forall(axis,wff):
            return 1/tf.reduce_mean(1/(wff+1e-10),axis=axis)

    if aggreg == "min":
        def F_Forall(axis,wff):
            return tf.reduce_min(wff,axis=axis)

    if aggreg == "mean":
        def F_Forall(axis,wff):
            return tf.reduce_mean(wff, axis=axis)


def set_existential_aggregator(aggreg):
    assert  aggreg in ['max']
    global F_Exists
    if aggreg == "max":
        def F_Exists(axis, wff):
            return tf.reduce_max(wff, axis=axis)


def And(*wffs):
    if len(wffs) == 0:
        result = tf.constant(1.0)
        result.doms = []
    else:
        cross_wffs,_ = cross_args(wffs)
        label = "_AND_".join([wff.name.split(":")[0] for wff in wffs])
        result = tf.identity(F_And(cross_wffs),name=label)
        result.doms = cross_wffs.doms
    return result


def Or(*wffs):
    if len(wffs) == 0:
        result = tf.constant(0.0)
        result.doms = []
    else:
        cross_wffs,_ = cross_args(wffs)
        label = "_OR_".join([wff.name.split(":")[0] for wff in wffs])
        result = tf.identity(F_Or(cross_wffs),name=label)
        result.doms = cross_wffs.doms
    return result


def Implies(wff1, wff2):
    _, cross_wffs = cross_2args(wff1,wff2)
    label = wff1.name.split(":")[0] + "_IMP_" + wff2.name.split(":")[0]
    result = F_Implies(cross_wffs[0],cross_wffs[1])
    result = tf.identity(result,name=label)
    result.doms = cross_wffs[0].doms
    return result


def Not(wff):
    result = F_Not(wff)
    label = "NOT_" + wff.name.split(":")[0]
    result = tf.identity(result,name=label)
    result.doms = wff.doms
    return result


def Equiv(wff1,wff2):
    _, cross_wffs = cross_2args(wff1,wff2)
    label = wff1.name.split(":")[0] + "_IFF_" + wff2.name.split(":")[0]
    result = F_Equiv(cross_wffs[0],cross_wffs[1])
    result.doms = cross_wffs[0].doms
    return result


def Forall(vars,wff):
    if type(vars) is not tuple:
        vars = (vars,)
    result_doms = [x for x in wff.doms if x not in [var.doms[0] for var in vars]]
    quantif_axis = [wff.doms.index(var.doms[0]) for var in vars]
    not_empty_vars = tf.cast(tf.reduce_prod(tf.stack([tf.size(var) for var in vars])),tf.bool)
    ones = tf.ones((1,)*(len(result_doms)+1))
    result = tf.cond(not_empty_vars,lambda:F_Forall(quantif_axis,wff),lambda:ones)
    result.doms = result_doms
    return result


def Exists(vars,wff):
    if type(vars) is not tuple:
        vars = (vars,)
    result_doms = [x for x in wff.doms if x not in [var.doms[0] for var in vars]]
    quantif_axis = [wff.doms.index(var.doms[0]) for var in vars]
    not_empty_vars = tf.cast(tf.reduce_prod(tf.stack([tf.size(var) for var in vars])),tf.bool)
    zeros = tf.zeros((1,)*(len(result_doms)+1))
    result = tf.cond(not_empty_vars,lambda:F_Exists(quantif_axis,wff),lambda:zeros)
    result.doms = result_doms
    return result


set_tnorm(DEFAULT_TNORM)
set_universal_aggreg(DEFAULT_UNIVERSAL_AGG)
set_existential_aggregator(DEFAULT_EXISTENTIAL_AGGREGATOR)