import numpy as np
import pandas as pd

import six
import tensorflow as tf
import edward as ed

from edward.inferences.variational_inference import VariationalInference
import edward as ed
from edward.models import Bernoulli, Normal, Multinomial, Categorical, Mixture, PointMass, Empirical
from edward.models import RandomVariable
from edward.util import copy, get_descendants
from tensorflow.contrib.distributions import kl_divergence
# tf.contrib.distributions.kl_divergence = tf.contrib.distributions.kl




def preprocess(data, colnames, continuous = True):
    data = data[data.APPRDX!=9]
    data = data.dropna()
    K = len(colnames)

    ## Generate gatherInd, X and t
    nrep = np.array(data.groupby('PATNO')['PATNO'].count())
    N = len(data['PATNO'].unique())
    gatherInd = np.repeat(np.arange(N),nrep)

    X_tmp = data.groupby('PATNO')['GENDER','bl_age','APPRDX'].first()
    ## Relabel gender: 0(Female of child bearing potential), 1(Female of non-child bearing potential), 2(Male)
    X_tmp['GENDER'] -= 1
    X_std = np.array(X_tmp[['GENDER']])
    X_std = np.hstack([np.ones([X_std.shape[0],1]),X_std])
    ## Combine female of child bearing and female of non-child bearing
    X_std[X_std[:,1]==-1,1] = 0
    
    t_mat = np.tile([np.array(data['age']),],(K,1)).T
    t_mat /= 60
    
    Y = np.array(data[colnames])
    Y = Y.astype('float32')
    if continuous:
        Y = (Y-np.mean(Y,0))/np.std(Y,0)
    return Y, gatherInd, X_std, t_mat, data
            
# Customized class inherited from Edward KLqp class            
class KLqp_customized(ed.inferences.KLpq):
  def initialize(self, L, sumsubL, n_samples=1, kl_scaling=None, *args, **kwargs):
    """Initialize inference algorithm. It initializes hyperparameters
    and builds ops for the algorithm's computation graph.
    Args:
      n_samples: int.
        Number of samples from variational model for calculating
        stochastic gradients.
      kl_scaling: dict of RandomVariable to tf.Tensor.
        Provides option to scale terms when using ELBO with KL divergence.
        If the KL divergence terms are
        $\\alpha_p \mathbb{E}_{q(z\mid x, \lambda)} [
              \log q(z\mid x, \lambda) - \log p(z)],$
        then pass {$p(z)$: $\\alpha_p$} as `kl_scaling`,
        where $\\alpha_p$ is a tensor. Its shape must be broadcastable;
        it is multiplied element-wise to the batchwise KL terms.
    """
    if kl_scaling is None:
      kl_scaling = {}
    if n_samples <= 0:
      raise ValueError(
          "n_samples should be greater than zero: {}".format(n_samples))

    self.n_samples = n_samples
    self.L = L
    self.sumsubL = sumsubL
    self.kl_scaling = kl_scaling
    return super(KLqp_customized, self).initialize(*args, **kwargs)

  def build_loss_and_gradients(self, var_list):
    return build_reparam_kl_loss_and_gradients_customized(self, var_list)

def build_reparam_kl_loss_and_gradients_customized(inference, var_list):
  """Build loss function. Its automatic differentiation
  is a stochastic gradient of
  .. math::
    -\\text{ELBO} =  - ( \mathbb{E}_{q(z; \lambda)} [ \log p(x \mid z) ]
          + \\text{KL}(q(z; \lambda) \| p(z)) )
  based on the reparameterization trick [@kingma2014auto].
  It assumes the KL is analytic.
  Computed by sampling from $q(z;\lambda)$ and evaluating the
  expectation using Monte Carlo sampling.
  """
  p_log_lik = [0.0] * inference.n_samples
  base_scope = tf.get_default_graph().unique_name("inference") + '/'
  for s in range(inference.n_samples):
    # Form dictionary in order to replace conditioning on prior or
    # observed variable with conditioning on a specific value.
    scope = base_scope + tf.get_default_graph().unique_name("sample")
    dict_swap = {}
    for x, qx in six.iteritems(inference.data):
      if isinstance(x, RandomVariable):
        if isinstance(qx, RandomVariable):
          qx_copy = copy(qx, scope=scope)
          dict_swap[x] = qx_copy.value()
        else:
          dict_swap[x] = qx

    for z, qz in six.iteritems(inference.latent_vars):
      # Copy q(z) to obtain new set of posterior samples.
      qz_copy = copy(qz, scope=scope)
      dict_swap[z] = qz_copy.value()

    for x in six.iterkeys(inference.data):
      if isinstance(x, RandomVariable):
        x_copy = copy(x, dict_swap, scope=scope)
        p_log_lik[s] += tf.reduce_sum(
            inference.scale.get(x, 1.0) * x_copy.log_prob(dict_swap[x]))

  p_log_lik = tf.reduce_mean(p_log_lik)

  kl_penalty = tf.reduce_sum([
      tf.reduce_sum(inference.kl_scaling.get(z, 1.0) * kl_divergence(qz, z))
      for z, qz in six.iteritems(inference.latent_vars)])

  cov_penalty = 0
  for i in range(inference.L+inference.sumsubL):
    for j in range(i+1,(inference.L+inference.sumsubL)):
      cov_tmp = tf.cast(1 / (tf.shape(qz.mean()[:,i])[0] - 1), tf.float32) * tf.reduce_sum((qz.mean()[:,i] - tf.reduce_mean(qz.mean()[:,i])) * (qz.mean()[:,j] - tf.reduce_mean(qz.mean()[:,j])))
      cov_penalty += tf.abs(cov_tmp)

  reg_penalty = tf.reduce_sum(tf.losses.get_regularization_losses())

  if inference.logging:
    tf.summary.scalar("loss/p_log_lik", p_log_lik,
                      collections=[inference._summary_key])
    tf.summary.scalar("loss/kl_penalty", kl_penalty,
                      collections=[inference._summary_key])
    tf.summary.scalar("loss/reg_penalty", reg_penalty,
                      collections=[inference._summary_key])
    
# Modified loss function to ensure 0 correlations among latent factors
  loss = -(p_log_lik - kl_penalty - reg_penalty - 10000*cov_penalty)

  grads = tf.gradients(loss, var_list)
  grads_and_vars = list(zip(grads, var_list))
  return loss, grads_and_vars



def add_offset(x, offset):
    return (x + offset) / (1 + offset)

# Build the MEFF model
def Model_building(L, Lsub, N, K0, K1, joinpid, data0, data1, gatherInd0, gatherInd1, t_mat0, t_mat1, X_std0, X_std1, Y0_dat, Y1_dat, b0posind, p1=2, C=4):
    offset = tf.constant(1e-8)
    sess = tf.Session()
    p2 = p1 - 1
    eta0 = [tf.get_variable('eta0'+str(i+1),[p1,K0[i]], dtype=tf.float32) for i in np.arange(len(K0))]
    b0 = [-tf.nn.softplus(tf.get_variable('b0'+str(i+1),[K0[i]], dtype = tf.float32)) for i in np.arange(len(K0))]
    mu0 = [tf.exp(tf.get_variable('mu0'+str(i+1),[K0[i]], tf.float32)) for i in np.arange(len(K0))]
    theta0 = [tf.get_variable('theta0'+str(i+1),[p2,K0[i]]) for i in np.arange(len(K0))]
    delta2 = [tf.get_variable('delta2k'+str(i+1),[K0[i]], dtype = tf.float32) for i in np.arange(len(K0))]
    delta3 = [tf.get_variable('delta3k'+str(i+1),[K0[i]], dtype = tf.float32) for i in np.arange(len(K0))]
    
    Klist = K0+K1
    Wtmp1 = np.array([[-np.infty]*L + [-np.infty]*Lsub[0] + [0]*sum(Lsub[1:])]*Klist[0])
    for i in range(1,len(Lsub)):
        arrtmp = [-np.infty]*L + [0]*sum(Lsub)
        arrtmp[(L+sum(Lsub[:i])):(L+sum(Lsub[:(i+1)]))] = [-np.infty]*Lsub[i]
        Wtmp1 = np.vstack([Wtmp1, np.array([arrtmp]*Klist[i])])
    Wtmp2 = np.array([[np.infty]*L + [np.infty]*Lsub[0] + [0]*sum(Lsub[1:])]*Klist[0])
    for i in range(1,len(Lsub)):
        arrtmp = [np.infty]*L + [0]*sum(Lsub)
        arrtmp[(L+sum(Lsub[:i])):(L+sum(Lsub[:(i+1)]))] = [np.infty]*Lsub[i]
        Wtmp2 = np.vstack([Wtmp2, np.array([arrtmp]*Klist[i])])
    W = tf.get_variable('W', [sum(K0)+sum(K1),L+sum(Lsub)],
                       constraint = lambda x: tf.clip_by_value(x,
                                                              tf.cast(Wtmp1, tf.float32),
                                                               tf.cast(Wtmp2, tf.float32)))
    
    u_tot = Normal(loc = tf.zeros([N, L+sum(Lsub)]), scale=tf.ones([N, L+sum(Lsub)]))
    W_tot = tf.matmul(u_tot, tf.transpose(W))

    ind0 = []; ind1 = []; N0 = []; N1 = []
    for i in range(len(K0)):
        ind = np.where(np.isin(joinpid, data0[i].PATNO))[0].astype(np.int32)
        ind0.append(ind)
        N0.append(len(ind))
    for i in range(len(K1)):
        ind = np.where(np.isin(joinpid, data1[i].PATNO))[0]
        ind1.append(ind)
        N1.append(len(ind))
    
    if len(K0) > 0:
        X_std_ph0 = [tf.placeholder(tf.float32,[N0[i],p1]) for i in range(len(K0))]
        X_eta0 = [tf.matmul(tf.cast(X_std_ph0[i], tf.float32), tf.cast(eta0[i], tf.float32)) for i in range(len(K0))]
        prob_Q0 = [add_offset(tf.sigmoid(X_eta0[i]), offset)  for i in range(len(K0))]
        prob_notQ0 = [add_offset(1-prob_Q0[i], offset) for i in range(len(K0))]
        prob_Q_ex0 = [tf.expand_dims(prob_Q0[i], 2) + offset for i in range(len(K0))]
        prob_notQ_ex0 = [add_offset(tf.expand_dims(prob_notQ0[i], 2), offset) for i in range(len(K0))]
        prob_Qcat0 = [tf.concat([prob_Q_ex0[i], prob_notQ_ex0[i]], -1) for i in range(len(K0))]
        prob_Qcat_gather0 = [tf.gather(prob_Qcat0[i], tf.cast(gatherInd0[i], tf.int32)) for i in range(len(K0))]

        W0 = [tf.gather(W_tot[:,0:K0[0]],ind0[0])] + [tf.gather(W_tot[:,np.cumsum(K0)[i]:np.cumsum(K0)[i+1]], ind0[i+1]) for i in range(0, len(K0)-1)]
        d0 = [mu0[i] + tf.matmul(tf.cast(X_std_ph0[i][:,1:p1], tf.float32), theta0[i]) + tf.cast(W0[i], tf.float32) for i in range(len(K0))]
        d_gather0 = [tf.gather(d0[i], tf.cast(gatherInd0[i], tf.int32)) for i in range(len(K0))]
        linear0 = [b0[i]*(tf.cast(t_mat0[i][:,0:K0[i]],tf.float32)-d_gather0[i]) for i in range(len(K0))]
        pro10 = [tf.expand_dims(3.*linear0[i]+delta2[i]+delta3[i], 2) for i in range(len(K0))]
        pro20 = [tf.expand_dims(2.*linear0[i]+delta2[i]+delta3[i], 2) for i in range(len(K0))]
        pro30 = [tf.expand_dims(linear0[i]+delta3[i], 2) for i in range(len(K0))]
        pro40 = [tf.cast(tf.expand_dims(tf.zeros([tf.shape(pro10[i])[0], tf.shape(pro10[i])[1]]), 2), tf.float32) for i in range(len(K0))]
        pro0 = [tf.concat([pro10[i], pro20[i], pro30[i], pro40[i]], -1) for i in range(len(K0))]
        H0 = [Categorical(logits = add_offset(pro0[i], offset)) for i in range(len(K0))]
        Qcat0 = [Categorical(probs=add_offset(prob_Qcat_gather0[i], offset)) for i in range(len(K0))]
        comps0 = [[Bernoulli(probs=add_offset(tf.zeros([gatherInd0[i].shape[0],K0[i]]), offset)),
                 H0[i]] for i in range(len(K0))]
        Y0 = [Mixture(cat=Qcat0[i], components=comps0[i]) for i in range(len(K0))]
        
    if len(K1) > 0:
        eta1 = [tf.get_variable('eta'+str(i+1),[p1,K1[i]], dtype=tf.float32) for i in np.arange(len(K1))]
        b1 = []
        for i in np.arange(len(K1)):
            if i==b0posind:
                b1 += [tf.nn.softplus(tf.get_variable('b'+str(i+1),[K1[b0posind]], dtype = tf.float32))]
            else:
                b1 += [-tf.nn.softplus(tf.get_variable('b'+str(i+1),[K1[i]], dtype = tf.float32))]
        mu1 = [tf.exp(tf.get_variable('mu'+str(i+1),[K1[i]], tf.float32)) for i in np.arange(len(K1))]
        theta1 = [tf.get_variable('theta'+str(i+1),[p2,K1[i]]) for i in np.arange(len(K1))]
        beta0 = [tf.get_variable('beta0'+str(i+1),[p1,K1[i]]) for i in np.arange(len(K1))]
        beta1 = [tf.get_variable('beta1'+str(i+1),[K1[i]]) for i in np.arange(len(K1))]
        sigma_ek = [tf.sqrt(tf.exp(tf.get_variable("sigma_ek"+str(i+1), [K1[i]]))) for i in np.arange(len(K1))]
        aest = [tf.nn.softplus(tf.get_variable('aest'+str(i+1),[K1[i]], dtype = tf.float32)) for i in np.arange(len(K1))]

        X_std_ph1 = [tf.placeholder(tf.float32,[N1[i],p1]) for i in np.arange(len(K1))]
        X_eta1 = [tf.matmul(tf.cast(X_std_ph1[i], tf.float32), tf.cast(eta1[i], tf.float32)) for i in np.arange(len(K1))]
        W1 = [tf.gather(W_tot[:,sum(K0):(sum(K0)+K1[0])],ind1[0])] + [tf.gather(W_tot[:,(sum(K0)+np.cumsum(K1)[i]):(sum(K0)+np.cumsum(K1)[i+1])], ind1[i+1]) for i in range(0,len(K1)-1)]
        d1 = [mu1[i] + tf.matmul(X_std_ph1[i][:,1:p1], theta1[i])+ W1[i] for i in range(len(K1))]
        d_gather1 = [tf.gather(d1[i], gatherInd1[i]) for i in range(len(K1))]
        linear1 = [b1[i]*(t_mat1[i][:,0:K1[i]]-d_gather1[i]) for i in range(len(K1))]

        Xbeta0 = [tf.matmul(tf.cast(X_std_ph1[i], tf.float32), beta0[i]) for i in range(len(K1))]
        Xbeta0_gather = [tf.gather(Xbeta0[i], gatherInd1[i]) for i in range(len(K1))]

        fixed = [Xbeta0_gather[i] + t_mat1[i][:,0:K1[i]]*beta1[i] + aest[i]*(1-1/add_offset((1+tf.gather(tf.exp(X_eta1[i]), gatherInd1[i]))*(1+tf.exp(linear1[i])), offset))
                for i in range(len(K1))]

        Y1 = [Normal(loc=fixed[i], scale=sigma_ek[i]*tf.ones([len(gatherInd1[i]),K1[i]])) for i in range(len(K1))]

    if len(K0)>0:
        datadict = {X_std_ph0[i]: X_std0[i] for i in range(len(K0))}
        datadict.update({Y0[i]:Y0_dat[i] for i in range(len(K0))})
    else:
        datadict = {}
    datadict1a = {X_std_ph1[i]: X_std1[i] for i in range(len(K1))}
    datadict1b = {Y1[i]: Y1_dat[i] for i in range(len(K1))}
    datadict.update(datadict1a)
    datadict.update(datadict1b)
    
    param_list = {}
    param_list["W"] = W
    param_list["u"] = u_tot

    if len(K0) > 0:
        param_list["theta0"] = theta0
        param_list["mu0"] = mu0
        param_list["b0"] = b0
        param_list["eta0"] = eta0
        param_list["delta2"] = delta2
        param_list["delta3"] = delta3

    if len(K1) > 0:
        param_list["eta1"] = eta1
        param_list["b1"] = b1
        param_list["mu1"] = mu1
        param_list["theta1"] = theta1
        param_list["beta0"] = beta0
        param_list["beta1"] = beta1
        param_list["sigma_ek"] = sigma_ek
        param_list["aest"] = aest
    return u_tot, datadict, param_list