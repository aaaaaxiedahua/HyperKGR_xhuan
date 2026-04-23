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

    def forward(self, q_sub, q_rel, hidden, path_state, edges, nodes, old_nodes_new_idx, batchsize,
                curvature, path_rel, query_path_proto, lambda_q):
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

        
        # sample edges w.r.t. alpha
        if self.n_edge_topk > 0:
            alpha          = self.w_alpha(nn.ReLU()(self.Ws_attn(hs) + self.Wr_attn(hr) + self.Wqr_attn(h_qr))).squeeze(-1)
            edge_prob      = F.gumbel_softmax(alpha, tau=1, hard=False)
            topk_index     = torch.argsort(edge_prob, descending=True)[:self.n_edge_topk]
            edge_prob_hard = torch.zeros((alpha.shape[0])).cuda()
            edge_prob_hard[topk_index] = 1
            alpha *= (edge_prob_hard - edge_prob.detach() + edge_prob)
            alpha = torch.sigmoid(alpha).unsqueeze(-1)
            
        else:
            alpha = torch.sigmoid(self.w_alpha(nn.ReLU()(self.Ws_attn(hs) + self.Wr_attn(hr) + self.Wqr_attn(h_qr)))) # [N_edge_of_all_batch, 1]
        

        # suppose all embedding are in tangetn space
        hr = expmap0(hr, curvature)
        hs = expmap0(hs, curvature)

        message = project(mobius_add(hs, hr, curvature), curvature) # hyperbolic space, hyperbolic transE
        #message = mobius_add(hs, hr, 1) # hyperbolic space, hyperbolic transE
        message = logmap0(message, curvature) # to tangent space
        #message = hs + hr


        # aggregate message and then propagate
        message     = alpha * message
        message_agg = scatter(message, index=obj, dim=0, dim_size=n_node, reduce='sum')

        # to poincare space
        a__ = self.W_h(message_agg)
        #a__ = p_exp_map(a__)
        a__ = expmap0(a__, curvature)
        hidden_new = self.act(a__)
        hidden_new = logmap0(hidden_new, curvature)

        #hidden_new  = self.act(self.W_h(message_agg)) # [n_node, dim]
        hidden_new  = hidden_new.clone()

        # Tangent-space path token update: consensus is built in Euclidean space,
        # then only the final readout will map it into the hyperbolic space.
        path_parent = path_state[sub]
        self_loop_mask = rel == (self.n_rel * 2)
        if self_loop_mask.any():
            path_rel = path_rel.clone()
            path_rel[self_loop_mask] = 0.0

        path_candidates = sanitize_tensor(path_parent + path_rel, nan=0.0)
        query_path_edge = query_path_proto[r_idx]
        path_query_sim = F.cosine_similarity(path_candidates, query_path_edge, dim=-1)
        path_logits = torch.log(alpha.squeeze(-1).clamp_min(MIN_NORM)) + lambda_q * path_query_sim
        path_logits = sanitize_tensor(path_logits, nan=-MAX_LOGIT, posinf=MAX_LOGIT, neginf=-MAX_LOGIT, min_value=-MAX_LOGIT, max_value=MAX_LOGIT)
        path_weight_max = scatter(path_logits, index=obj, dim=0, dim_size=n_node, reduce='max')
        path_weight_exp = torch.exp(path_logits - path_weight_max[obj])
        path_weight_sum = scatter(path_weight_exp, index=obj, dim=0, dim_size=n_node, reduce='sum')
        path_weights = path_weight_exp / path_weight_sum[obj].clamp_min(MIN_NORM)
        path_weights = sanitize_tensor(path_weights, nan=0.0, posinf=1.0, neginf=0.0)

        path_state_new = scatter(
            path_weights.unsqueeze(-1) * path_candidates,
            index=obj,
            dim=0,
            dim_size=n_node,
            reduce='sum',
        )
        path_dispersion = scatter(
            path_weights * (path_candidates - path_state_new[obj]).pow(2).sum(dim=-1),
            index=obj,
            dim=0,
            dim_size=n_node,
            reduce='sum',
        )
        path_state_new = sanitize_tensor(path_state_new, nan=0.0)
        path_dispersion = sanitize_tensor(path_dispersion, nan=0.0, posinf=MAX_DISTANCE, neginf=0.0, min_value=0.0, max_value=MAX_DISTANCE)
        
        # forward without node sampling
        if self.n_node_topk <= 0:
            keep_mask = torch.ones(n_node, dtype=torch.bool, device=hidden_new.device)
            return hidden_new, path_state_new, path_dispersion, nodes, keep_mask

        # forward with node sampling
        # indexing sampling operation
        tmp_diff_node_idx = torch.ones(n_node, device=hidden_new.device)
        tmp_diff_node_idx[old_nodes_new_idx] = 0
        bool_diff_node_idx = tmp_diff_node_idx.bool()
        diff_node = nodes[bool_diff_node_idx]

        # project logit to fixed-size tensor via indexing
        diff_node_logit  = self.W_samp(hidden_new[bool_diff_node_idx]).squeeze(-1) # [all_batch_new_nodes]
        
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
        path_state_new = path_state_new[bool_same_node_idx]
        path_dispersion = path_dispersion[bool_same_node_idx]

        return hidden_new, path_state_new, path_dispersion, new_nodes, bool_same_node_idx

class GNNModel(torch.nn.Module):
    def __init__(self, params, loader):
        super(GNNModel, self).__init__()
        self.n_layer     = params.n_layer
        self.hidden_dim  = params.hidden_dim
        self.d_path      = params.d_path
        self.d_type      = params.d_type
        self.attn_dim    = params.attn_dim
        self.n_ent       = params.n_ent
        self.n_rel       = params.n_rel
        self.n_node_topk = params.n_node_topk
        self.n_edge_topk = params.n_edge_topk
        self.loader      = loader
        self.lambda_q    = params.lambda_q
        self.lambda_cal  = params.lambda_cal
        self.beta_delta  = params.beta_delta
        self.eta         = params.eta
        self.b_g         = params.b_g
        self.mu_t        = params.mu_t
        self.gamma_t     = params.gamma_t
        acts = {'relu': nn.ReLU(), 'tanh': torch.tanh, 'idd': lambda x:x}
        act  = acts[params.act]
        self.curvature = torch.nn.Parameter(torch.tensor(1.0))

        self.gnn_layers = []
        for i in range(self.n_layer):
            i_n_node_topk = self.n_node_topk if 'int' in str(type(self.n_node_topk)) else self.n_node_topk[i]
            self.gnn_layers.append(GNNLayer(self.hidden_dim, self.hidden_dim, self.attn_dim, self.n_rel, self.n_ent, self.d_path, \
                                            n_node_topk=i_n_node_topk, n_edge_topk=self.n_edge_topk, tau=params.tau, act=act))

        self.gnn_layers = nn.ModuleList(self.gnn_layers)       
        self.dropout = nn.Dropout(params.dropout)
        self.W_final = nn.Linear(self.hidden_dim, 1, bias=False)
        self.gate    = nn.GRU(self.hidden_dim, self.hidden_dim)
        self.path_rela_embed = nn.Embedding(2 * self.n_rel + 1, self.d_path)
        self.path_prototypes = nn.Embedding(2 * self.n_rel, self.d_path)
        self.type_prototypes = nn.Embedding(2 * self.n_rel, self.d_type)
        self.W_path = nn.Linear(self.d_path, self.d_path, bias=False)
        self.W_type = nn.Linear(self.hidden_dim, self.d_type, bias=False)

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
        path_state = torch.zeros(n, self.d_path).cuda()                                   # [B, d_path] tangent-space path token
    
        for i in range(self.n_layer):
            # layers with sampling
            # nodes (of i-th layer): [k1, 2]
            # edges (of i-th layer): [k2, 6]
            # old_nodes_new_idx (of previous layer): [k1']
            nodes, edges, old_nodes_new_idx = self.loader.get_neighbors(nodes.data.cpu().numpy(), n, mode=mode)
            n_node  = nodes.size(0)
            query_path_proto = self.path_prototypes(q_rel)
            path_rel = self.path_rela_embed(edges[:,2])

            # GNN forward -> get hidden representation at i-th layer
            # hidden: [k1, dim]
            hidden, path_state, path_dispersion, nodes, sampled_nodes_idx = self.gnn_layers[i](
                q_sub, q_rel, hidden, path_state, edges, nodes, old_nodes_new_idx, n,
                self.curvature, path_rel, query_path_proto, self.lambda_q
            )

            # combine h0 and hi -> update hi with gate operation
            h0          = torch.zeros(1, n_node, hidden.size(1)).cuda().index_copy_(1, old_nodes_new_idx, h0)
            h0          = h0[0, sampled_nodes_idx, :].unsqueeze(0)
            hidden      = self.dropout(hidden)
            hidden, h0  = self.gate(hidden.unsqueeze(0), h0)
            hidden      = hidden.squeeze(0)
            
        # readout
        # [K, 2] (batch_idx, node_idx) K is w.r.t. n_nodes
        scores_base = sanitize_tensor(self.W_final(hidden).squeeze(-1), nan=0.0)
        node_query_rel = q_rel[nodes[:,0]]
        path_state_h = expmap0(self.W_path(path_state), self.curvature)
        query_path_proto = expmap0(self.W_path(self.path_prototypes(node_query_rel)), self.curvature)
        path_scores = -hyp_distance(path_state_h, query_path_proto, self.curvature, eval_mode=False).squeeze(-1) / max(self.eta, 1e-6)
        path_scores = sanitize_tensor(path_scores, nan=-MAX_DISTANCE, posinf=0.0, neginf=-MAX_DISTANCE, min_value=-MAX_DISTANCE, max_value=0.0)
        type_hidden = sanitize_tensor(self.W_type(hidden), nan=0.0)
        type_scores = F.cosine_similarity(type_hidden, self.type_prototypes(node_query_rel), dim=-1)
        type_scores = sanitize_tensor(type_scores, nan=0.0, posinf=1.0, neginf=-1.0, min_value=-1.0, max_value=1.0)
        gate = torch.sigmoid(self.b_g - self.beta_delta * path_dispersion)
        gate = sanitize_tensor(gate, nan=0.5, posinf=1.0, neginf=0.0, min_value=0.0, max_value=1.0)
        calibrated_scores = sanitize_tensor(gate * path_scores + (1 - gate) * type_scores, nan=0.0)
        scores = sanitize_tensor(scores_base + self.lambda_cal * calibrated_scores, nan=-MAX_DISTANCE)
        # non-visited entities have 0 scores
        scores_all = torch.zeros((n, self.loader.n_ent)).cuda()         
        # [B, n_all_nodes]
        scores_all[[nodes[:,0], nodes[:,1]]] = scores                   
        if not return_details:
            return scores_all

        return scores_all, {
            'nodes': nodes,
            'hidden': hidden,
            'type_hidden': type_hidden,
            'path_state': path_state,
            'path_state_h': path_state_h,
            'path_dispersion': path_dispersion,
            'q_rel': q_rel,
        }

    def auxiliary_losses(self, details, rels, targets):
        nodes = details['nodes']
        hidden = details['hidden']
        type_hidden = details['type_hidden']
        q_rel = torch.as_tensor(rels, dtype=torch.long, device=hidden.device)
        targets = torch.as_tensor(targets, dtype=torch.long, device=hidden.device)
        batch_size = q_rel.size(0)

        node_lookup = torch.full((batch_size, self.n_ent), -1, dtype=torch.long, device=hidden.device)
        node_lookup[nodes[:,0], nodes[:,1]] = torch.arange(nodes.size(0), device=hidden.device)
        pos_idx = node_lookup[torch.arange(batch_size, device=hidden.device), targets]
        valid_mask = pos_idx >= 0
        if valid_mask.sum() <= 1:
            return hidden.new_tensor(0.0)

        pos_idx = pos_idx[valid_mask]
        rel_idx = q_rel[valid_mask]
        pos_type = type_hidden[pos_idx]
        type_proto = self.type_prototypes(rel_idx)
        type_logits = (F.normalize(type_proto, dim=-1) @ F.normalize(pos_type, dim=-1).transpose(0, 1)) / max(self.gamma_t, 1e-6)
        type_logits = sanitize_tensor(type_logits, nan=0.0, posinf=MAX_LOGIT, neginf=-MAX_LOGIT, min_value=-MAX_LOGIT, max_value=MAX_LOGIT)
        labels = torch.arange(pos_idx.size(0), device=hidden.device)
        return F.cross_entropy(type_logits, labels)
