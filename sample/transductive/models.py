import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import numpy as np
from   torch_scatter import scatter
from   collections import defaultdict




# Define constants to avoid problematic values
MIN_CURVATURE = 1e-6  # Minimum allowed value for curvature c

def safe_curvature(c):
    """Ensure curvature c is not zero or too small."""
    return c.clamp_min(MIN_CURVATURE)


def mobius_addition(x, y, *, c=1.0):
    c = safe_curvature(c)
    c = torch.as_tensor(c).type_as(x)
    return _mobius_add(x, y, c)

def _mobius_add(x, y, c):
    c = safe_curvature(c)
    x2 = x.pow(2).sum(dim=-1, keepdim=True)
    y2 = y.pow(2).sum(dim=-1, keepdim=True)
    xy = (x * y).sum(dim=-1, keepdim=True)
    num = (1 + 2 * c * xy + c * y2) * x + (1 - c * x2) * y
    denom = 1 + 2 * c * xy + c ** 2 * x2 * y2
    return num / (denom + 1e-5)



def exp_map(x, v, c=1.0):
    c = safe_curvature(c)
    norm_v = torch.norm(v, dim=-1, keepdim=True).clamp(min=1e-8)
    return x + torch.tanh(c * norm_v) * (v / norm_v)

def log_map(x, y, c=1.0):
    c = safe_curvature(c)
    diff = y - x
    norm_diff = torch.norm(diff, dim=-1, keepdim=True).clamp(min=1e-8)
    return (1 / c) * torch.atanh(c * norm_diff) * (diff / norm_diff)

def hyperbolic_distance(x, y, c=1.0):
    c = safe_curvature(c)
    diff = x - y
    norm_diff = torch.norm(diff, dim=-1, keepdim=True).clamp(min=1e-8)
    return (2 / c) * torch.atanh(c * norm_diff)



def artanh(x):
    return 0.5*torch.log((1+x)/(1-x))

def p_exp_map(v):
    normv = torch.clamp(torch.norm(v, 2, dim=-1, keepdim=True), min=1e-10)
    return torch.tanh(normv)*v/normv

def p_log_map(v):
    normv = torch.clamp(torch.norm(v, 2, dim=-1, keepdim=True), 1e-10, 1-1e-5)
    return artanh(normv)*v/normv

def full_p_exp_map(x, v):
    normv = torch.clamp(torch.norm(v, 2, dim=-1, keepdim=True), min=1e-10)
    sqxnorm = torch.clamp(torch.sum(x * x, dim=-1, keepdim=True), 0, 1-1e-5)
    y = torch.tanh(normv/(1-sqxnorm)) * v/normv
    return p_sum(x, y)

def p_sum(x, y):
    sqxnorm = torch.clamp(torch.sum(x * x, dim=-1, keepdim=True), 0, 1-1e-5)
    sqynorm = torch.clamp(torch.sum(y * y, dim=-1, keepdim=True), 0, 1-1e-5)
    dotxy = torch.sum(x*y, dim=-1, keepdim=True)
    numerator = (1+2*dotxy+sqynorm)*x + (1-sqxnorm)*y
    denominator = 1 + 2*dotxy + sqxnorm*sqynorm
    return numerator/denominator


import torch

MIN_NORM = 1e-15
BALL_EPS = {torch.float32: 4e-3, torch.float64: 1e-5}


# ################# MATH FUNCTIONS ########################

class Artanh(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        x = x.clamp(-1 + 1e-5, 1 - 1e-5)
        ctx.save_for_backward(x)
        dtype = x.dtype
        x = x.double()
        return (torch.log_(1 + x).sub_(torch.log_(1 - x))).mul_(0.5).to(dtype)

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        return grad_output / (1 - input ** 2)


def artanh(x):
    return Artanh.apply(x)


def tanh(x):
    return x.clamp(-15, 15).tanh()


# ################# HYP OPS ########################

def expmap0(u, c=1):
    """Exponential map taken at the origin of the Poincare ball with curvature c.

    Args:
        u: torch.Tensor of size B x d with hyperbolic points
        c: torch.Tensor of size 1 or B x 1 with absolute hyperbolic curvatures

    Returns:
        torch.Tensor with tangent points.
    """
    c = safe_curvature(c)
    sqrt_c = c ** 0.5
    u_norm = u.norm(dim=-1, p=2, keepdim=True).clamp_min(MIN_NORM)
    gamma_1 = tanh(sqrt_c * u_norm) * u / (sqrt_c * u_norm)
    return project(gamma_1, c)


def logmap0(y, c):
    """Logarithmic map taken at the origin of the Poincare ball with curvature c.

    Args:
        y: torch.Tensor of size B x d with tangent points
        c: torch.Tensor of size 1 or B x 1 with absolute hyperbolic curvatures

    Returns:
        torch.Tensor with hyperbolic points.
    """
    c = safe_curvature(c)
    sqrt_c = c ** 0.5
    y_norm = y.norm(dim=-1, p=2, keepdim=True).clamp_min(MIN_NORM)
    return y / y_norm / sqrt_c * artanh(sqrt_c * y_norm)


def project(x, c):
    """Project points to Poincare ball with curvature c.

    Args:
        x: torch.Tensor of size B x d with hyperbolic points
        c: torch.Tensor of size 1 or B x 1 with absolute hyperbolic curvatures

    Returns:
        torch.Tensor with projected hyperbolic points.
    """
    c = safe_curvature(c)
    norm = x.norm(dim=-1, p=2, keepdim=True).clamp_min(MIN_NORM)
    eps = BALL_EPS[x.dtype]
    maxnorm = (1 - eps) / (c ** 0.5)
    cond = norm > maxnorm
    projected = x / norm * maxnorm
    return torch.where(cond, projected, x)


def mobius_add(x, y, c):
    """Mobius addition of points in the Poincare ball with curvature c.

    Args:
        x: torch.Tensor of size B x d with hyperbolic points
        y: torch.Tensor of size B x d with hyperbolic points
        c: torch.Tensor of size 1 or B x 1 with absolute hyperbolic curvatures

    Returns:
        Tensor of shape B x d representing the element-wise Mobius addition of x and y.
    """
    c = safe_curvature(c)
    x2 = torch.sum(x * x, dim=-1, keepdim=True)
    y2 = torch.sum(y * y, dim=-1, keepdim=True)
    xy = torch.sum(x * y, dim=-1, keepdim=True)
    num = (1 + 2 * c * xy + c * y2) * x + (1 - c * x2) * y
    denom = 1 + 2 * c * xy + c ** 2 * x2 * y2
    return num / denom.clamp_min(MIN_NORM)


def mobius_scalar_mul(r, x, c):
    """Mobius scalar multiplication on the Poincare ball."""
    c = safe_curvature(c)
    sqrt_c = c ** 0.5
    x_norm = x.norm(dim=-1, p=2, keepdim=True).clamp_min(MIN_NORM)
    return tanh(r * artanh(sqrt_c * x_norm)) * x / (sqrt_c * x_norm)


# ################# HYP DISTANCES ########################

def hyp_distance(x, y, c, eval_mode=False):
    """Hyperbolic distance on the Poincare ball with curvature c.

    Args:
        x: torch.Tensor of size B x d with hyperbolic queries
        y: torch.Tensor with hyperbolic queries, shape n_entities x d if eval_mode is true else (B x d)
        c: torch.Tensor of size 1 with absolute hyperbolic curvature

    Returns: torch,Tensor with hyperbolic distances, size B x 1 if eval_mode is False
            else B x n_entities matrix with all pairs distances
    """
    c = safe_curvature(c)
    sqrt_c = c ** 0.5
    x2 = torch.sum(x * x, dim=-1, keepdim=True)
    if eval_mode:
        y2 = torch.sum(y * y, dim=-1, keepdim=True).transpose(0, 1)
        xy = x @ y.transpose(0, 1)
    else:
        y2 = torch.sum(y * y, dim=-1, keepdim=True)
        xy = torch.sum(x * y, dim=-1, keepdim=True)
    c1 = 1 - 2 * c * xy + c * y2
    c2 = 1 - c * x2
    num = torch.sqrt((c1 ** 2) * x2 + (c2 ** 2) * y2 - (2 * c1 * c2) * xy)
    denom = 1 - 2 * c * xy + c ** 2 * x2 * y2
    pairwise_norm = num / denom.clamp_min(MIN_NORM)
    dist = artanh(sqrt_c * pairwise_norm)
    return 2 * dist / sqrt_c


def hyp_distance_multi_c(x, v, c, eval_mode=False):
    """Hyperbolic distance on Poincare balls with varying curvatures c.

    Args:
        x: torch.Tensor of size B x d with hyperbolic queries
        y: torch.Tensor with hyperbolic queries, shape n_entities x d if eval_mode is true else (B x d)
        c: torch.Tensor of size B x d with absolute hyperbolic curvatures

    Return: torch,Tensor with hyperbolic distances, size B x 1 if eval_mode is False
            else B x n_entities matrix with all pairs distances
    """
    c = safe_curvature(c)
    sqrt_c = c ** 0.5
    if eval_mode:
        vnorm = torch.norm(v, p=2, dim=-1, keepdim=True).transpose(0, 1)
        xv = x @ v.transpose(0, 1) / vnorm
    else:
        vnorm = torch.norm(v, p=2, dim=-1, keepdim=True)
        xv = torch.sum(x * v / vnorm, dim=-1, keepdim=True)
    gamma = tanh(sqrt_c * vnorm) / sqrt_c
    x2 = torch.sum(x * x, dim=-1, keepdim=True)
    c1 = 1 - 2 * c * gamma * xv + c * gamma ** 2
    c2 = 1 - c * x2
    num = torch.sqrt((c1 ** 2) * x2 + (c2 ** 2) * (gamma ** 2) - (2 * c1 * c2) * gamma * xv)
    denom = 1 - 2 * c * gamma * xv + (c ** 2) * (gamma ** 2) * x2
    pairwise_norm = num / denom.clamp_min(MIN_NORM)
    dist = artanh(sqrt_c * pairwise_norm)
    return 2 * dist / sqrt_c

##########################################################################################################################################################################################

class GNNLayer(torch.nn.Module):
    def __init__(self, in_dim, out_dim, attn_dim, n_rel, n_ent, d_path, n_node_topk=-1, n_edge_topk=-1, tau=1.0, act=lambda x:x):
        super(GNNLayer, self).__init__()
        self.n_rel       = n_rel
        self.n_ent       = n_ent
        self.in_dim      = in_dim
        self.out_dim     = out_dim
        self.attn_dim    = attn_dim
        self.d_path      = d_path
        self.act         = act
        self.n_node_topk = n_node_topk
        self.n_edge_topk = n_edge_topk
        self.tau         = tau
        self.rela_embed  = nn.Embedding(2*n_rel+1, in_dim)
        self.Ws_attn     = nn.Linear(in_dim, attn_dim, bias=False)
        self.Wr_attn     = nn.Linear(in_dim, attn_dim, bias=False)
        self.Wqr_attn    = nn.Linear(in_dim, attn_dim)
        self.w_alpha     = nn.Linear(attn_dim, 1)
        self.W_h         = nn.Linear(in_dim, out_dim, bias=False)
        self.W_samp      = nn.Linear(in_dim, 1, bias=False)
        self.W_path_prev = nn.Linear(d_path, d_path, bias=False)
        self.W_path_rel  = nn.Linear(in_dim, d_path, bias=False)

        # if the dataset is NELL, make it not changable
        self.curvature = torch.nn.Parameter(torch.tensor(1.0))

    def train(self, mode=True):
        if not isinstance(mode, bool):
            raise ValueError("training mode is expected to be boolean")
        self.training = mode
        if self.training and self.tau > 0:
            self.softmax = lambda x : F.gumbel_softmax(x, tau=self.tau, hard=False)
        else:
            self.softmax = lambda x : F.softmax(x, dim=1)
        for module in self.children():
            module.train(mode)
        return self

    def forward(self, q_sub, q_rel, hidden, path_state, edges, nodes, old_nodes_new_idx, batchsize):
        # edges: [N_edge_of_all_batch, 6]
        # with (batch_idx, head, rela, tail, head_idx, tail_idx)
        sub    = edges[:,4]
        rel    = edges[:,2]
        obj    = edges[:,5]
        hs     = hidden[sub]
        hp     = path_state[sub]
        hr     = self.rela_embed(rel)
        r_idx  = edges[:,0]
        h_qr   = self.rela_embed(q_rel)[r_idx]
        n_node = nodes.shape[0]

        attn_hidden = nn.ReLU()(self.Ws_attn(hs) + self.Wr_attn(hr) + self.Wqr_attn(h_qr))
        attn_logits = self.w_alpha(attn_hidden).squeeze(-1)
        if self.n_edge_topk > 0:
            edge_prob      = F.gumbel_softmax(attn_logits, tau=1, hard=False)
            topk_index     = torch.argsort(edge_prob, descending=True)[:self.n_edge_topk]
            edge_prob_hard = torch.zeros_like(attn_logits)
            edge_prob_hard[topk_index] = 1
            attn_logits *= (edge_prob_hard - edge_prob.detach() + edge_prob)
            alpha = torch.sigmoid(attn_logits).unsqueeze(-1)
        else:
            alpha = torch.sigmoid(attn_logits).unsqueeze(-1)

        path_edge = torch.tanh(self.W_path_prev(hp) + self.W_path_rel(hr))

        # suppose all embedding are in tangent space
        hr = expmap0(hr, self.curvature)
        hs = expmap0(hs, self.curvature)

        message = project(mobius_add(hs, hr, self.curvature), self.curvature)
        message = logmap0(message, self.curvature)

        message_agg = scatter(alpha * message, index=obj, dim=0, dim_size=n_node, reduce='sum')
        path_state_new = scatter(alpha * path_edge, index=obj, dim=0, dim_size=n_node, reduce='sum')

        a__ = self.W_h(message_agg)
        a__ = expmap0(a__, self.curvature)
        hidden_new = self.act(a__)
        hidden_new = logmap0(hidden_new, self.curvature)

        hidden_new = hidden_new.clone()
        path_state_new = path_state_new.clone()

        if self.n_node_topk <= 0:
            return hidden_new, path_state_new

        tmp_diff_node_idx = torch.ones(n_node, device=hidden_new.device)
        tmp_diff_node_idx[old_nodes_new_idx] = 0
        bool_diff_node_idx = tmp_diff_node_idx.bool()
        diff_node = nodes[bool_diff_node_idx]

        diff_node_logit = self.W_samp(hidden_new[bool_diff_node_idx]).squeeze(-1)

        node_scores = torch.ones((batchsize, self.n_ent), device=hidden_new.device) * float('-inf')
        node_scores[diff_node[:,0], diff_node[:,1]] = diff_node_logit

        node_scores = self.softmax(node_scores)
        topk_index = torch.topk(node_scores, self.n_node_topk, dim=1).indices.reshape(-1)
        topk_batchidx = torch.arange(batchsize, device=hidden_new.device).repeat(self.n_node_topk, 1).T.reshape(-1)
        batch_topk_nodes = torch.zeros((batchsize, self.n_ent), device=hidden_new.device)
        batch_topk_nodes[topk_batchidx, topk_index] = 1

        bool_sampled_diff_nodes_idx = batch_topk_nodes[diff_node[:,0], diff_node[:,1]].bool()
        bool_same_node_idx = ~bool_diff_node_idx
        bool_same_node_idx[bool_diff_node_idx] = bool_sampled_diff_nodes_idx

        diff_node_prob_hard = batch_topk_nodes[diff_node[:,0], diff_node[:,1]]
        diff_node_prob = node_scores[diff_node[:,0], diff_node[:,1]]
        sample_gate = (diff_node_prob_hard - diff_node_prob.detach() + diff_node_prob).unsqueeze(-1)
        hidden_new[bool_diff_node_idx] *= sample_gate
        path_state_new[bool_diff_node_idx] *= sample_gate

        new_nodes = nodes[bool_same_node_idx]
        hidden_new = hidden_new[bool_same_node_idx]
        path_state_new = path_state_new[bool_same_node_idx]

        return hidden_new, path_state_new, new_nodes, bool_same_node_idx


class GNNModel(torch.nn.Module):
    def __init__(self, params, loader):
        super(GNNModel, self).__init__()
        self.n_layer     = params.n_layer
        self.hidden_dim  = params.hidden_dim
        self.attn_dim    = params.attn_dim
        self.d_path      = params.d_path
        self.d_score     = params.d_score
        self.n_ent       = params.n_ent
        self.n_rel       = params.n_rel
        self.n_node_topk = params.n_node_topk
        self.n_edge_topk = params.n_edge_topk
        self.loader      = loader
        acts = {'relu': nn.ReLU(), 'tanh': torch.tanh, 'idd': lambda x:x}
        act  = acts[params.act]

        self.gnn_layers = []
        for i in range(self.n_layer):
            i_n_node_topk = self.n_node_topk if 'int' in str(type(self.n_node_topk)) else self.n_node_topk[i]
            self.gnn_layers.append(GNNLayer(
                self.hidden_dim,
                self.hidden_dim,
                self.attn_dim,
                self.n_rel,
                self.n_ent,
                self.d_path,
                n_node_topk=i_n_node_topk,
                n_edge_topk=self.n_edge_topk,
                tau=params.tau,
                act=act,
            ))

        self.gnn_layers = nn.ModuleList(self.gnn_layers)
        self.dropout = nn.Dropout(params.dropout)
        self.gate = nn.GRU(self.hidden_dim, self.hidden_dim)

        self.query_rela_embed = nn.Embedding(2 * self.n_rel + 1, self.hidden_dim)
        self.path_init = nn.Linear(self.hidden_dim, self.d_path, bias=False)
        self.W_path_fuse = nn.Linear(self.d_path, self.hidden_dim, bias=False)
        self.W_score_node = nn.Linear(self.hidden_dim, self.d_score, bias=False)
        self.W_score_rel = nn.Linear(self.hidden_dim, self.d_score, bias=False)
        self.W_final = nn.Linear(self.hidden_dim, 1, bias=False)
        self.score_curvature = nn.Parameter(torch.tensor(1.0))
        self.score_scale = max(float(self.d_score), 1.0) ** 0.5

    def updateTopkNums(self, topk_list):
        assert len(topk_list) == self.n_layer
        for idx in range(self.n_layer):
            self.gnn_layers[idx].n_node_topk = topk_list[idx]

    def fixSamplingWeight(self):
        def freeze(m):
            m.requires_grad=False
        for i in range(self.n_layer):
            self.gnn_layers[i].W_samp.apply(freeze)

    def path_hyperbolic_fuse(self, hidden, path_state):
        path_proj = torch.tanh(self.W_path_fuse(path_state))
        hidden_hyp = expmap0(hidden, self.score_curvature)
        path_hyp = expmap0(path_proj, self.score_curvature)
        fused_hyp = project(mobius_add(hidden_hyp, path_hyp, self.score_curvature), self.score_curvature)
        return logmap0(fused_hyp, self.score_curvature)

    def relation_aware_hscore(self, fused_hidden, nodes, q_rel):
        batch_idx = nodes[:,0]
        node_tangent = torch.tanh(self.W_score_node(fused_hidden))
        rel_tangent = torch.tanh(self.W_score_rel(self.query_rela_embed(q_rel)))[batch_idx]
        node_hyp = expmap0(node_tangent, self.score_curvature)
        rel_hyp = expmap0(rel_tangent, self.score_curvature)
        dist = hyp_distance(node_hyp, rel_hyp, self.score_curvature, eval_mode=False).squeeze(-1)
        return -dist / self.score_scale

    def forward(self, subs, rels, mode='train'):
        n = len(subs)
        q_sub = torch.LongTensor(subs).cuda()
        q_rel = torch.LongTensor(rels).cuda()
        h0 = torch.zeros((1, n, self.hidden_dim)).cuda()
        nodes = torch.cat([torch.arange(n).unsqueeze(1).cuda(), q_sub.unsqueeze(1)], 1)
        hidden = torch.zeros(n, self.hidden_dim).cuda()
        path_state = torch.tanh(self.path_init(self.query_rela_embed(q_rel)))

        for i in range(self.n_layer):
            nodes, edges, old_nodes_new_idx = self.loader.get_neighbors(nodes.data.cpu().numpy(), n, mode=mode)
            n_node = nodes.size(0)

            layer_out = self.gnn_layers[i](q_sub, q_rel, hidden, path_state, edges, nodes, old_nodes_new_idx, n)
            if len(layer_out) == 2:
                hidden, path_state = layer_out
                sampled_nodes_idx = torch.arange(hidden.size(0), device=hidden.device)
            else:
                hidden, path_state, nodes, sampled_nodes_idx = layer_out

            h0 = torch.zeros(1, n_node, hidden.size(1), device=hidden.device).index_copy_(1, old_nodes_new_idx, h0)
            h0 = h0[0, sampled_nodes_idx, :].unsqueeze(0)
            hidden = self.dropout(hidden)
            hidden, h0 = self.gate(hidden.unsqueeze(0), h0)
            hidden = hidden.squeeze(0)

        fused_hidden = self.path_hyperbolic_fuse(hidden, path_state)
        base_scores = self.W_final(fused_hidden).squeeze(-1)
        hyp_scores = self.relation_aware_hscore(fused_hidden, nodes, q_rel)
        scores = base_scores + hyp_scores
        scores_all = torch.zeros((n, self.loader.n_ent), device=hidden.device)
        scores_all[[nodes[:,0], nodes[:,1]]] = scores

        return scores_all
