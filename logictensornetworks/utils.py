import tensorflow as tf


def cross_args(args):
    result = args[0]
    for arg in args[1:]:
        result,_ = cross_2args(result,arg)
    result_flat = tf.reshape(result,
                             (tf.reduce_prod(tf.shape(result)[:-1]),
                              tf.shape(result)[-1]))
    result_args = tf.split(result_flat,[tf.shape(arg)[-1] for arg in args],1)
    return result, result_args


def cross_2args(X,Y):
    if X.doms == [] and Y.doms == []:
        result = tf.concat([X,Y],axis=-1)
        result.doms = []
        return result,[X,Y]
    X_Y = set(X.doms) - set(Y.doms)
    Y_X = set(Y.doms) - set(X.doms)
    eX = X
    eX_doms = [x for x in X.doms]
    for y in Y_X:
        eX = tf.expand_dims(eX,0)
        eX_doms.append(y)
    eY = Y
    eY_doms = [y for y in Y.doms]
    for x in X_Y:
        eY = tf.expand_dims(eY,0)
        eY_doms.append(x)
    perm_eY = []
    for y in eY_doms:
        perm_eY.append(eX_doms.index(y))
    eY = tf.transpose(eY,perm_eY + [len(perm_eY)])
    mult_eX = [1]*(len(eX_doms)+1)
    mult_eY = [1]*(len(eY_doms)+1)
    for i in range(len(mult_eX)-1):
        mult_eX[i] = tf.maximum(1,tf.floor_div(tf.shape(eY)[i],tf.shape(eX)[i]))
        mult_eY[i] = tf.maximum(1,tf.floor_div(tf.shape(eX)[i],tf.shape(eY)[i]))
    result1 = tf.tile(eX,mult_eX)
    result2 = tf.tile(eY,mult_eY)
    result = tf.concat([result1,result2],axis=-1)
    result1.doms = eX_doms
    result2.doms = eX_doms
    result.doms = eX_doms
    return result,[result1,result2]