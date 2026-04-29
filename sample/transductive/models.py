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
MAX_DISTANCE = 1e6
MAX_LOGIT = 50.0


def sanitize_tensor(x, nan=0.0, posinf=MAX_DISTANCE, neginf=-MAX_DISTANCE, min_value=None, max_value=None):
    x = torch.nan_to_num(x, nan=nan, posinf=posinf, neginf=neginf)
    if min_value is not None or max_value is not None:
        x = torch.clamp(x, min=min_value, max=max_value)
    return x


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
    return project(sanitize_tensor(gamma_1), c)


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
    result = y / y_norm / sqrt_c * artanh((sqrt_c * y_norm).clamp_max(1 - BALL_EPS[y.dtype]))
    return sanitize_tensor(result)


def project(x, c):
    """Project points to Poincare ball with curvature c.

    Args:
        x: torch.Tensor of size B x d with hyperbolic points
        c: torch.Tensor of size 1 or B x 1 with absolute hyperbolic curvatures

    Returns:
        torch.Tensor with projected hyperbolic points.
    """
    c = safe_curvature(c)
    x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    norm = x.norm(dim=-1, p=2, keepdim=True).clamp_min(MIN_NORM)
    eps = BALL_EPS[x.dtype]
    maxnorm = (1 - eps) / (c ** 0.5)
    cond = norm > maxnorm
    projected = x / norm * maxnorm
    return sanitize_tensor(torch.where(cond, projected, x), nan=0.0, posinf=0.0, neginf=0.0)


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
    return project(sanitize_tensor(num / denom.clamp_min(MIN_NORM)), c)


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
    radicand = (c1 ** 2) * x2 + (c2 ** 2) * y2 - (2 * c1 * c2) * xy
    num = torch.sqrt(radicand.clamp_min(0.0))
    denom = 1 - 2 * c * xy + c ** 2 * x2 * y2
    pairwise_norm = sanitize_tensor(num / denom.clamp_min(MIN_NORM), nan=0.0, posinf=MAX_DISTANCE, neginf=0.0, min_value=0.0)
    max_norm = (1 - BALL_EPS[x.dtype]) / sqrt_c.clamp_min(MIN_NORM)
    pairwise_norm = torch.minimum(pairwise_norm, max_norm)
    dist = artanh((sqrt_c * pairwise_norm).clamp_max(1 - BALL_EPS[x.dtype]))
    return sanitize_tensor(2 * dist / sqrt_c, nan=0.0, posinf=MAX_DISTANCE, neginf=0.0, min_value=0.0, max_value=MAX_DISTANCE)


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
    def __init__(self, in_dim, out_dim, attn_dim, n_rel, n_ent, d_rule, n_node_topk=-1, n_edge_topk=-1, tau=1.0, act=lambda x:x):
        super(GNNLayer, self).__init__()
        self.n_rel       = n_rel
        self.n_ent       = n_ent
        self.in_dim      = in_dim
        self.out_dim     = out_dim
        self.attn_dim    = attn_dim
        self.d_rule      = d_rule
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
        self.rule_attn   = nn.Linear(d_rule, attn_dim, bias=True)
        self.rule_msg    = nn.Linear(d_rule, 1, bias=True)
        nn.init.zeros_(self.rule_attn.weight)
        nn.init.zeros_(self.rule_attn.bias)
        nn.init.zeros_(self.rule_msg.weight)
        nn.init.zeros_(self.rule_msg.bias)
        
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

    def forward(self, q_sub, q_rel, hidden, edges, nodes, old_nodes_new_idx, batchsize,
                curvature, edge_rule, query_rule_pref):
        # edges: [N_edge_of_all_batch, 6] 
        # with (batch_idx, head, rela, tail, head_idx, tail_idx)
        # note that head_idx and tail_idx are relative index
        sub     = edges[:,4]
        rel     = edges[:,2]
        obj     = edges[:,5]
        hs      = hidden[sub]
        hr      = self.rela_embed(rel)
        r_idx   = edges[:,0]
        h_qr    = self.rela_embed(q_rel)[r_idx]
        n_node  = nodes.shape[0]

        
        base_attn_feat = self.Ws_attn(hs) + self.Wr_attn(hr) + self.Wqr_attn(h_qr)
        query_rule_edge = query_rule_pref[r_idx]
        rule_context = edge_rule * query_rule_edge
        self_loop_mask = rel == (self.n_rel * 2)
        if self_loop_mask.any():
            rule_context = rule_context.clone()
            rule_context[self_loop_mask] = 0.0
        rule_context = sanitize_tensor(rule_context, nan=0.0, posinf=1.0, neginf=-1.0, min_value=-1.0, max_value=1.0)
        attn_scale = 2.0 * torch.sigmoid(self.rule_attn(rule_context))
        attn_scale = sanitize_tensor(attn_scale, nan=1.0, posinf=2.0, neginf=0.0, min_value=0.0, max_value=2.0)
        attn_feat = attn_scale * base_attn_feat
        alpha_logit = sanitize_tensor(
            self.w_alpha(nn.ReLU()(attn_feat)).squeeze(-1),
            nan=-MAX_LOGIT,
            posinf=MAX_LOGIT,
            neginf=-MAX_LOGIT,
            min_value=-MAX_LOGIT,
            max_value=MAX_LOGIT,
        )

        # sample edges w.r.t. rule-aware alpha
        if self.n_edge_topk > 0:
            edge_prob      = F.gumbel_softmax(alpha_logit, tau=1, hard=False)
            topk_index     = torch.argsort(edge_prob, descending=True)[:self.n_edge_topk]
            edge_prob_hard = torch.zeros((alpha_logit.shape[0])).cuda()
            edge_prob_hard[topk_index] = 1
            alpha_logit = alpha_logit * (edge_prob_hard - edge_prob.detach() + edge_prob)
            
        alpha_max = scatter(alpha_logit, index=obj, dim=0, dim_size=n_node, reduce='max')
        alpha_exp = torch.exp(alpha_logit - alpha_max[obj])
        alpha_sum = scatter(alpha_exp, index=obj, dim=0, dim_size=n_node, reduce='sum')
        alpha = alpha_exp / alpha_sum[obj].clamp_min(MIN_NORM)
        alpha = sanitize_tensor(alpha, nan=0.0, posinf=1.0, neginf=0.0, min_value=0.0, max_value=1.0)

        # suppose all embedding are in tangetn space
        hr = expmap0(hr, curvature)
        hs = expmap0(hs, curvature)

        message = project(mobius_add(hs, hr, curvature), curvature) # hyperbolic space, hyperbolic transE
        #message = mobius_add(hs, hr, 1) # hyperbolic space, hyperbolic transE
        message = logmap0(message, curvature) # to tangent space
        #message = hs + hr


        # aggregate message and then propagate
        rule_gate   = 2.0 * torch.sigmoid(self.rule_msg(rule_context).squeeze(-1))
        rule_gate   = sanitize_tensor(rule_gate, nan=1.0, posinf=2.0, neginf=0.0, min_value=0.0, max_value=2.0)
        message     = alpha.unsqueeze(-1) * rule_gate.unsqueeze(-1) * message
        message_agg = scatter(message, index=obj, dim=0, dim_size=n_node, reduce='sum')

        # to poincare space
        a__ = self.W_h(message_agg)
        #a__ = p_exp_map(a__)
        a__ = expmap0(a__, curvature)
        hidden_new = self.act(a__)
        hidden_new = logmap0(hidden_new, curvature)

        #hidden_new  = self.act(self.W_h(message_agg)) # [n_node, dim]
        hidden_new  = hidden_new.clone()

        # forward without node sampling
        if self.n_node_topk <= 0:
            keep_mask = torch.ones(n_node, dtype=torch.bool, device=hidden_new.device)
            return hidden_new, nodes, keep_mask

        # forward with node sampling
        # indexing sampling operation
        tmp_diff_node_idx = torch.ones(n_node, device=hidden_new.device)
        tmp_diff_node_idx[old_nodes_new_idx] = 0
        bool_diff_node_idx = tmp_diff_node_idx.bool()
        diff_node = nodes[bool_diff_node_idx]

        # project logit to fixed-size tensor via indexing
        diff_node_logit = self.W_samp(hidden_new[bool_diff_node_idx]).squeeze(-1) # [all_batch_new_nodes]
        
        # save logit to node_scores for later indexing
        node_scores = torch.ones((batchsize, self.n_ent)).cuda() * float('-inf')
        node_scores[diff_node[:,0], diff_node[:,1]] = diff_node_logit

        # select top-k nodes
        # (train mode) self.softmax == F.gumbel_softmax
        # (eval mode)  self.softmax == F.softmax 
        node_scores = self.softmax(node_scores) # [batchsize, n_ent]
        topk_index = torch.topk(node_scores, self.n_node_topk, dim=1).indices.reshape(-1)
        topk_batchidx  = torch.arange(batchsize).repeat(self.n_node_topk,1).T.reshape(-1)
        batch_topk_nodes = torch.zeros((batchsize, self.n_ent)).cuda()
        batch_topk_nodes[topk_batchidx, topk_index] = 1

        # get sampled nodes' relative index
        bool_sampled_diff_nodes_idx = batch_topk_nodes[diff_node[:,0], diff_node[:,1]].bool()
        bool_same_node_idx = ~bool_diff_node_idx.cuda()
        bool_same_node_idx[bool_diff_node_idx] = bool_sampled_diff_nodes_idx

        # update node embeddings
        diff_node_prob_hard = batch_topk_nodes[diff_node[:,0], diff_node[:,1]]
        diff_node_prob = node_scores[diff_node[:,0], diff_node[:,1]]
        hidden_new[bool_diff_node_idx] *= (diff_node_prob_hard - diff_node_prob.detach() + diff_node_prob).unsqueeze(-1)
        
        # extract sampled nodes an their embeddings
        new_nodes  = nodes[bool_same_node_idx]
        hidden_new = hidden_new[bool_same_node_idx]

        return hidden_new, new_nodes, bool_same_node_idx

class GNNModel(torch.nn.Module):
    def __init__(self, params, loader):
        super(GNNModel, self).__init__()
        self.n_layer     = params.n_layer
        self.hidden_dim  = params.hidden_dim
        self.d_rule      = params.d_rule
        self.d_buffer    = params.d_buffer
        self.attn_dim    = params.attn_dim
        self.n_ent       = params.n_ent
        self.n_rel       = params.n_rel
        self.n_node_topk = params.n_node_topk
        self.n_edge_topk = params.n_edge_topk
        self.loader      = loader
        acts = {'relu': nn.ReLU(), 'tanh': torch.tanh, 'idd': lambda x:x}
        act  = acts[params.act]
        self.curvature = torch.nn.Parameter(torch.tensor(1.0))

        self.gnn_layers = []
        for i in range(self.n_layer):
            i_n_node_topk = self.n_node_topk if 'int' in str(type(self.n_node_topk)) else self.n_node_topk[i]
            self.gnn_layers.append(GNNLayer(self.hidden_dim, self.hidden_dim, self.attn_dim, self.n_rel, self.n_ent, self.d_rule, \
                                            n_node_topk=i_n_node_topk, n_edge_topk=self.n_edge_topk, tau=params.tau, act=act))

        self.gnn_layers = nn.ModuleList(self.gnn_layers)       
        self.dropout = nn.Dropout(params.dropout)
        self.W_final = nn.Linear(self.hidden_dim, 1, bias=False)
        self.rule_embed = nn.Embedding(2 * self.n_rel + 1, self.d_rule)
        self.rule_query = nn.ModuleList([nn.Linear(self.d_rule, self.d_rule, bias=False) for _ in range(self.n_layer)])
        self.buffer_dropout = nn.Dropout(params.buffer_dropout)
        self.buffer_mlp = nn.Linear(2 * self.hidden_dim, self.d_buffer)
        self.buffer_out = nn.Linear(self.d_buffer, self.hidden_dim)
        nn.init.zeros_(self.buffer_mlp.bias)
        nn.init.zeros_(self.buffer_out.bias)

    def buffer_update(self, hidden_new, hidden_old):
        buffer_feat = torch.cat([hidden_new, hidden_old], dim=-1)
        gate = torch.sigmoid(self.buffer_out(self.buffer_dropout(F.relu(self.buffer_mlp(buffer_feat)))))
        hidden = gate * hidden_new + (1.0 - gate) * hidden_old
        return sanitize_tensor(hidden, nan=0.0, posinf=1.0, neginf=-1.0)

    def updateTopkNums(self, topk_list):
        assert len(topk_list) == self.n_layer
        for idx in range(self.n_layer):
            self.gnn_layers[idx].n_node_topk = topk_list[idx]

    def fixSamplingWeight(self):
        def freeze(m):
            m.requires_grad=False
        for i in range(self.n_layer):
            self.gnn_layers[i].W_samp.apply(freeze)

    def forward(self, subs, rels, mode='train', return_details=False):
        n      = len(subs)                                                                # n == B (Batchsize)
        q_sub  = torch.LongTensor(subs).cuda()                                            # [B]
        q_rel  = torch.LongTensor(rels).cuda()                                            # [B]
        h0     = torch.zeros((1, n, self.hidden_dim)).cuda()                              # [1, B, dim]
        nodes  = torch.cat([torch.arange(n).unsqueeze(1).cuda(), q_sub.unsqueeze(1)], 1)  # [B, 2] with (batch_idx, node_idx)
        hidden = torch.zeros(n, self.hidden_dim).cuda()                                   # [B, dim]
    
        for i in range(self.n_layer):
            # layers with sampling
            # nodes (of i-th layer): [k1, 2]
            # edges (of i-th layer): [k2, 6]
            # old_nodes_new_idx (of previous layer): [k1']
            nodes, edges, old_nodes_new_idx = self.loader.get_neighbors(nodes.data.cpu().numpy(), n, mode=mode)
            n_node  = nodes.size(0)
            edge_rule = self.rule_embed(edges[:,2])
            query_rule_pref = self.rule_query[i](self.rule_embed(q_rel))

            # GNN forward -> get hidden representation at i-th layer
            # hidden: [k1, dim]
            hidden, nodes, sampled_nodes_idx = self.gnn_layers[i](
                q_sub, q_rel, hidden, edges, nodes, old_nodes_new_idx, n,
                self.curvature, edge_rule, query_rule_pref
            )

            # Lightweight buffer replaces GRU without adding recurrent CUDA state.
            h_old       = torch.zeros(1, n_node, hidden.size(1)).cuda().index_copy_(1, old_nodes_new_idx, h0)
            h_old       = h_old[0, sampled_nodes_idx, :]
            hidden      = self.dropout(hidden)
            hidden      = self.buffer_update(hidden, h_old)
            h0          = hidden.unsqueeze(0)
            
        # readout
        # [K, 2] (batch_idx, node_idx) K is w.r.t. n_nodes
        scores = sanitize_tensor(self.W_final(hidden).squeeze(-1), nan=-MAX_DISTANCE)
        # non-visited entities have 0 scores
        scores_all = torch.zeros((n, self.loader.n_ent)).cuda()         
        # [B, n_all_nodes]
        scores_all[[nodes[:,0], nodes[:,1]]] = scores                   
        if not return_details:
            return scores_all

        return scores_all, {
            'nodes': nodes,
            'hidden': hidden,
            'q_rel': q_rel,
        }
