#!/usr/bin/env python
import tensorflow as tf
from logictensornetworks.utils import cross_args

LAYERS = 4
BIAS_factor = 0.0
BIAS = 0.0


class Predicate:

    def __init__(self, label, number_of_features_or_vars, grounding_definition=None):
        self.label = label
        self.n_features = self._obtain_n_features(number_of_features_or_vars)

        if grounding_definition is None:
            self.W, self.u = self._declare_tensors_for_default_grounding()
            self.grounding_definition = self._default_grounding_definition
            self.pars = [self.W, self.u]
        else:
            self.grounding_definition = grounding_definition
            self.pars = []

    def _obtain_n_features(self, number_of_features_or_vars):
        if type(number_of_features_or_vars) is list:
            return sum([int(v.shape[1]) for v in number_of_features_or_vars])
        elif type(number_of_features_or_vars) is tf.Tensor:
            return int(number_of_features_or_vars.shape[1])
        else:
            return number_of_features_or_vars

    def _declare_tensors_for_default_grounding(self):
        """
        Default tensor shapes for use with _default_pred_definition
        For more details on these matrices, please see the method description for
        Predicate._default_pred_definition
        :return:
        """
        W = tf.matrix_band_part(
            tf.Variable(
                tf.random_normal(
                    [LAYERS,
                     self.n_features + 1,
                     self.n_features + 1], mean=0, stddev=1), name="W" + self.label), 0, -1)
        u = tf.Variable(tf.ones([LAYERS, 1]),
                        name="u" + self.label)
        return [W, u]

    def _default_grounding_definition(self, *args):
        """
        Default predicate definition.

        Serafini, Luciano, and Artur d’Avila Garcez. “Logic Tensor Networks: Deep Learning and Logical Reasoning from Data and Knowledge.” ArXiv:1606.04422 [Cs], June 14, 2016. http://arxiv.org/abs/1606.04422.

        "the grounding of the m-ary predicate P, G(P), is defined as a generalisation of the
        Neural Tensor Network [26] ... as a function from R^mn to [0,1] as follows:
        
        G(P) = sigmoid( u_P^T * tanh(v^T.W^[1:k]_P.v + V_P.v + B_p) )

        where W^[1:k]_P is a 3D tensor in R^mn*mn*k, V_P is a matrix in R^k*mn,
         and B_P is vector in R^k.

         With this encoding, the grounding (ie truth value, for predicates) of a clause can be determnined by
         a neural network which first computes the grounding of the literals contained in the clause,
         and then combines them using a specific s-norm.

        """

        # todo: find out why the bias is missing from this formula (compared to the paper)

        app_label = self.label + "/" + "_".join([arg.name.split(":")[0] for arg in args]) + "/"
        tensor_args = tf.concat(args, axis=1)
        X = tf.concat([tf.ones((tf.shape(tensor_args)[0], 1)),
                       tensor_args], 1)
        XW = tf.matmul(tf.tile(tf.expand_dims(X, 0), [LAYERS, 1, 1]), self.W)
        XWX = tf.squeeze(tf.matmul(tf.expand_dims(X, 1), tf.transpose(XW, [1, 2, 0])), axis=[1])
        gX = tf.matmul(tf.tanh(XWX), self.u)
        result = tf.sigmoid(gX, name=app_label)
        return result

    def ground(self, *args):

        pars = self.pars

        def predicate_grounding(*args):
            # global BIAS
            crossed_args, list_of_args_in_crossed_args = cross_args(args)
            result = self.grounding_definition(*list_of_args_in_crossed_args)
            if crossed_args.doms != []:
                result = tf.reshape(result, tf.concat([tf.shape(crossed_args)[:-1], [1]], axis=0))
            else:
                result = tf.reshape(result, (1,))
            result.doms = crossed_args.doms
            BIAS = get_bias()
            update_bias(tf.divide(BIAS + .5 - tf.reduce_mean(result), 2) * BIAS_factor)
            return result

        predicate_grounding.pars = pars
        predicate_grounding.label = self.label

        return predicate_grounding(*args)


class Function:

    def __init__(self, label, input_shape_spec, output_shape_spec=1,fun_definition=None):
        self.label = label
        self.n_features = self._obtain_n_features(input_shape_spec)
        self.output_shape_spec = output_shape_spec
        if fun_definition is None:
            self.W = self._declare_tensors_for_default_grounding()
            self.func_definition = self._default_grounding_definition
            self.pars = [self.W]
        else:
            self.func_definition = fun_definition
            self.pars = []

    def _obtain_n_features(self, input_shape_spec):
        if type(input_shape_spec) is list:
            return sum([int(v.shape[1]) for v in input_shape_spec])
        elif type(input_shape_spec) is tf.Tensor:
            return int(input_shape_spec.shape[1])
        else:
            return input_shape_spec

    def _declare_tensors_for_default_grounding(self):
        W = tf.Variable(
                tf.random_normal(
                    [self.n_features + 1,self.output_shape_spec],
                    mean=0,stddev=1
                ), name="W" + self.label)
        return W

    def _default_grounding_definition(self, *args):
        tensor_args = tf.concat(args, axis=1)
        X = tf.concat([tf.ones((tf.shape(tensor_args)[0], 1)),
                       tensor_args], 1)
        result = tf.matmul(X, self.W)
        return result

    def ground(self, *args):
        def function_grounding(*args):
            crossed_args, list_of_args_in_crossed_args = cross_args(args)
            result = self.func_definition(*list_of_args_in_crossed_args)
            if crossed_args.doms != []:
                result = tf.reshape(result, tf.concat([tf.shape(crossed_args)[:-1],
                                                       tf.shape(result)[-1:]], axis=0))
            else:
                result = tf.reshape(result, (self.output_shape_spec,))
            result.doms = crossed_args.doms
            return result

        function_grounding.pars = self.pars
        function_grounding.label = self.label
        return function_grounding(*args)


class Constant:

    def __init__(self, label, value=None, min_value=None, max_value=None):
        self.label = label
        self.value = value
        self.min_value = min_value
        self.max_value = max_value

    @property
    def ground(self):
        label = "ltn_constant_" + self.label
        if self.value is not None:
            result = tf.constant(self.value, name=label)
        else:
            result = tf.Variable(tf.random_uniform(
                shape=(1, len(self.min_value)),
                minval=self.min_value,
                maxval=self.max_value, name=label))
        result.doms = []
        return result

def predicate(label,number_of_features_or_vars,pred_definition=None):
    global BIAS
    if type(number_of_features_or_vars) is list:
        number_of_features = sum([int(v.shape[1]) for v in number_of_features_or_vars])
    elif type(number_of_features_or_vars) is tf.Tensor:
        number_of_features = int(number_of_features_or_vars.shape[1])
    else:
        number_of_features = number_of_features_or_vars
    if pred_definition is None:
        W = tf.matrix_band_part(
            tf.Variable(
                tf.random_normal(
                    [LAYERS,
                     number_of_features + 1,
                     number_of_features + 1],mean=0,stddev=1), name="W" + label), 0, -1)
        u = tf.Variable(tf.ones([LAYERS, 1]),
                        name="u" + label)
        def apply_pred(*args):
            app_label = label + "/" + "_".join([arg.name.split(":")[0] for arg in args]) + "/"
            tensor_args = tf.concat(args,axis=1)
            X = tf.concat([tf.ones((tf.shape(tensor_args)[0], 1)),
                           tensor_args], 1)
            XW = tf.matmul(tf.tile(tf.expand_dims(X, 0), [LAYERS, 1, 1]), W)
            XWX = tf.squeeze(tf.matmul(tf.expand_dims(X, 1), tf.transpose(XW, [1, 2, 0])), axis=[1])
            gX = tf.matmul(tf.tanh(XWX), u)
            result = tf.sigmoid(gX, name=app_label)
            return result
        pars = [W,u]
    else:
        def apply_pred(*args):
            return pred_definition(*args)
        pars = []

    def pred(*args):
        global BIAS
        crossed_args, list_of_args_in_crossed_args = cross_args(args)
        result = apply_pred(*list_of_args_in_crossed_args)
        if crossed_args.doms != []:
            result = tf.reshape(result, tf.concat([tf.shape(crossed_args)[:-1],[1]],axis=0))
        else:
            result = tf.reshape(result, (1,))
        result.doms = crossed_args.doms
        BIAS = tf.divide(BIAS + .5 - tf.reduce_mean(result),2)*BIAS_factor
        return result
    pred.pars = pars
    pred.label=label
    return pred

def function(label, input_shape_spec, output_shape_spec=1,fun_definition=None):
    if type(input_shape_spec) is list:
        number_of_features = sum([int(v.shape[1]) for v in input_shape_spec])
    elif type(input_shape_spec) is tf.Tensor:
        number_of_features = int(input_shape_spec.shape[1])
    else:
        number_of_features = input_shape_spec
    if fun_definition is None:
        W = tf.Variable(
                tf.random_normal(
                    [number_of_features + 1,output_shape_spec],mean=0,stddev=1), name="W" + label)
        def apply_fun(*args):
            tensor_args = tf.concat(args,axis=1)
            X = tf.concat([tf.ones((tf.shape(tensor_args)[0], 1)),
                           tensor_args], 1)
            result = tf.matmul(X,W)
            return result
        pars = [W]
    else:
        def apply_fun(*args):
            return fun_definition(*args)
        pars = []

    def fun(*args):
        crossed_args, list_of_args_in_crossed_args = cross_args(args)
        result = apply_fun(*list_of_args_in_crossed_args)
        if crossed_args.doms != []:
            result = tf.reshape(result, tf.concat([tf.shape(crossed_args)[:-1],
                                                   tf.shape(result)[-1:]],axis=0))
        else:
            result = tf.reshape(result, (output_shape_spec,))
        result.doms = crossed_args.doms
        return result
    fun.pars = pars
    fun.label=label
    return fun

def constant(label,value=None,
                 min_value=None,
                 max_value=None):
    label = "ltn_constant_"+label
    if value is not None:
        result = tf.constant(value,name=label)
    else:
        result = tf.Variable(tf.random_uniform(
                shape=(1,len(min_value)),
                minval=min_value,
                maxval=max_value,name=label))
    result.doms = []
    return result

def variable(label,number_of_features_or_feed):
    if type(number_of_features_or_feed) is int:
        result = tf.placeholder(dtype=tf.float32,shape=(None,number_of_features_or_feed),name=label)
    elif isinstance(number_of_features_or_feed,tf.Tensor):
        result = tf.identity(number_of_features_or_feed,name=label)
    else:
        result = tf.constant(number_of_features_or_feed,name=label)
    result.doms = [label]
    return result

def proposition(label,initial_value=None,value=None):
    if value is not None:
        assert 0 <= value and value <= 1
        result = tf.constant([value])
    elif initial_value is not None:
        assert 0 <= initial_value <= 1
        result = tf.Variable(initial_value=[value])
    else:
        result = tf.expand_dims(tf.clip_by_value(tf.Variable(tf.random_normal(shape=(),mean=.5,stddev=.5)),0.,1.),dim=0)
    result.doms = ()
    return result


def get_bias():
    global BIAS
    return BIAS

def update_bias(new_value):
    global BIAS
    BIAS = new_value
    return BIAS

def update_bias_factor(new_value):
    global BIAS_factor
    BIAS_factor = new_value
    return BIAS_factor


