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
    def __init__(
        self,
        in_dim,
        out_dim,
        attn_dim,
        n_rel,
        n_ent,
        n_node_topk=-1,
        n_edge_topk=-1,
        tau=1.0,
        act=lambda x:x,
        loader=None,
        shortcut_hops=1,
        shortcut_topk=0,
        shortcut_decay=0.5,
        shortcut_lambda=0.0,
        shortcut_candidate_cap=64,
        d_hop=None,
        shortcut_prune_lambda=-1.0,
    ):
        super(GNNLayer, self).__init__()
        self.n_rel       = n_rel
        self.n_ent       = n_ent
        self.in_dim      = in_dim
        self.out_dim     = out_dim
        self.attn_dim    = attn_dim
        self.act         = act
        self.n_node_topk = n_node_topk
        self.n_edge_topk = n_edge_topk
        self.tau         = tau
        self.loader      = loader
        self.rela_embed  = nn.Embedding(2*n_rel+1, in_dim)
        self.Ws_attn     = nn.Linear(in_dim, attn_dim, bias=False)
        self.Wr_attn     = nn.Linear(in_dim, attn_dim, bias=False)
        self.Wqr_attn    = nn.Linear(in_dim, attn_dim)
        self.w_alpha     = nn.Linear(attn_dim, 1)
        self.W_h         = nn.Linear(in_dim, out_dim, bias=False)
        self.W_samp      = nn.Linear(in_dim, 1, bias=False)
        self.shortcut_hops = max(1, int(shortcut_hops))
        self.shortcut_topk = int(shortcut_topk)
        self.shortcut_decay = float(shortcut_decay)
        self.shortcut_lambda = float(shortcut_lambda)
        self.shortcut_candidate_cap = int(shortcut_candidate_cap)
        self.d_hop = int(d_hop) if d_hop is not None and d_hop > 0 else out_dim
        if shortcut_prune_lambda is None or shortcut_prune_lambda < 0:
            self.shortcut_prune_lambda = self.shortcut_lambda
        else:
            self.shortcut_prune_lambda = float(shortcut_prune_lambda)
        self.use_shortcut = (
            self.loader is not None
            and self.shortcut_hops >= 2
            and self.shortcut_topk > 0
            and self.shortcut_candidate_cap != 0
            and (self.shortcut_lambda > 0 or self.shortcut_prune_lambda > 0)
        )
        if self.use_shortcut:
            self.hop_embed = nn.Embedding(self.shortcut_hops + 1, self.d_hop)
            self.W_sc_src = nn.Linear(out_dim, out_dim, bias=False)
            self.W_sc_dst = nn.Linear(out_dim, out_dim, bias=False)
            self.W_sc_query = nn.Linear(in_dim, out_dim, bias=False)
            self.W_sc_hop = nn.Linear(self.d_hop, out_dim, bias=False)
            self.w_sc_score = nn.Linear(out_dim, 1, bias=False)
            self.W_sc_msg = nn.Linear(out_dim, out_dim, bias=False)
            self.W_sc_hop_gate = nn.Linear(2 * out_dim, self.shortcut_hops - 1)
            self.W_sc_fuse = nn.Linear(3 * out_dim, out_dim)
            self.W_sc_prune = nn.Linear(3 * out_dim, 1, bias=False)


        # if the dataset is NELL, make it not changable
        self.curvature = torch.nn.Parameter(torch.tensor(1.0)) 
        #self.curvature = torch.tensor(1.0, requires_grad=False)

    def _shortcut_k(self, hop):
        decay = max(0.0, self.shortcut_decay)
        return max(1, int(round(self.shortcut_topk * (decay ** max(0, hop - 2)))))

    def _select_topk_per_source(self, src_idx, scores, k_h):
        selected = torch.zeros_like(scores, dtype=torch.bool)
        if src_idx.numel() == 0:
            return selected

        for src in torch.unique(src_idx):
            local_mask = src_idx == src
            local_count = int(local_mask.sum().item())
            if local_count <= k_h:
                selected[local_mask] = True
                continue
            local_positions = torch.nonzero(local_mask, as_tuple=False).view(-1)
            top_positions = torch.topk(scores[local_mask], k=k_h).indices
            selected[local_positions[top_positions]] = True
        return selected

    def _build_shortcut_context(self, hidden_new, nodes, q_rel, mode):
        if not self.use_shortcut or hidden_new.numel() == 0:
            return hidden_new, None

        shortcut_edges = self.loader.get_shortcut_edges(
            nodes,
            max_hops=self.shortcut_hops,
            candidate_cap=self.shortcut_candidate_cap,
            mode=mode,
        )
        if len(shortcut_edges) == 0:
            return hidden_new, None

        n_node = hidden_new.size(0)
        query_hidden = self.W_sc_query(self.rela_embed(q_rel[nodes[:, 0]]))
        hop_messages = []
        has_selected_edges = False

        for hop in range(2, self.shortcut_hops + 1):
            hop_message = hidden_new.new_zeros(n_node, self.out_dim)
            edge_index = shortcut_edges.get(hop)
            if edge_index is None:
                hop_messages.append(hop_message)
                continue

            src_np, dst_np = edge_index
            src_idx = torch.as_tensor(src_np, device=hidden_new.device, dtype=torch.long)
            dst_idx = torch.as_tensor(dst_np, device=hidden_new.device, dtype=torch.long)
            hop_ids = torch.full((src_idx.numel(),), hop, dtype=torch.long, device=hidden_new.device)

            edge_hidden = torch.tanh(
                self.W_sc_src(hidden_new[src_idx])
                + self.W_sc_dst(hidden_new[dst_idx])
                + query_hidden[src_idx]
                + self.W_sc_hop(self.hop_embed(hop_ids))
            )
            edge_scores = self.w_sc_score(edge_hidden).squeeze(-1)

            selected_mask = self._select_topk_per_source(src_idx, edge_scores, self._shortcut_k(hop))
            src_idx = src_idx[selected_mask]
            dst_idx = dst_idx[selected_mask]
            edge_scores = edge_scores[selected_mask]

            if src_idx.numel() == 0:
                hop_messages.append(hop_message)
                continue

            has_selected_edges = True
            edge_alpha = torch.zeros_like(edge_scores)
            for src in torch.unique(src_idx):
                local_mask = src_idx == src
                edge_alpha[local_mask] = F.softmax(edge_scores[local_mask], dim=0)

            hop_message = scatter(
                edge_alpha.unsqueeze(-1) * self.W_sc_msg(hidden_new[dst_idx]),
                index=src_idx,
                dim=0,
                dim_size=n_node,
                reduce='sum',
            )
            hop_messages.append(hop_message)

        if not has_selected_edges:
            return hidden_new, None

        hop_gate = torch.sigmoid(self.W_sc_hop_gate(torch.cat([hidden_new, query_hidden], dim=-1)))
        shortcut_message = hidden_new.new_zeros(n_node, self.out_dim)
        for hop_offset, hop_message in enumerate(hop_messages):
            shortcut_message = shortcut_message + hop_gate[:, hop_offset:hop_offset+1] * hop_message

        shortcut_features = torch.cat([hidden_new, shortcut_message, query_hidden], dim=-1)
        fuse_gate = torch.sigmoid(self.W_sc_fuse(shortcut_features))
        if self.shortcut_lambda > 0:
            hidden_new = hidden_new + fuse_gate * (self.shortcut_lambda * shortcut_message)

        shortcut_bonus = None
        if self.shortcut_prune_lambda > 0:
            shortcut_bonus = self.W_sc_prune(shortcut_features).squeeze(-1)
        return hidden_new, shortcut_bonus
        
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

    def forward(self, q_sub, q_rel, hidden, edges, nodes, old_nodes_new_idx, batchsize, mode='train'):
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
            edge_prob_hard = torch.zeros((alpha.shape[0]), device=alpha.device)
            edge_prob_hard[topk_index] = 1
            alpha *= (edge_prob_hard - edge_prob.detach() + edge_prob)
            alpha = torch.sigmoid(alpha).unsqueeze(-1)
            
        else:
            alpha = torch.sigmoid(self.w_alpha(nn.ReLU()(self.Ws_attn(hs) + self.Wr_attn(hr) + self.Wqr_attn(h_qr)))) # [N_edge_of_all_batch, 1]
        

        # suppose all embedding are in tangetn space
        hr = expmap0(hr, self.curvature)
        hs = expmap0(hs, self.curvature)

        message = project(mobius_add(hs, hr, self.curvature), self.curvature) # hyperbolic space, hyperbolic transE
        #message = mobius_add(hs, hr, 1) # hyperbolic space, hyperbolic transE
        message = logmap0(message, self.curvature) # to tangent space
        #message = hs + hr


        # aggregate message and then propagate
        message     = alpha * message
        message_agg = scatter(message, index=obj, dim=0, dim_size=n_node, reduce='sum')

        # to poincare space
        a__ = self.W_h(message_agg)
        #a__ = p_exp_map(a__)
        a__ = expmap0(a__, self.curvature)
        hidden_new = self.act(a__)
        hidden_new = logmap0(hidden_new, self.curvature)

        #hidden_new  = self.act(self.W_h(message_agg)) # [n_node, dim]
        hidden_new  = hidden_new.clone()
        hidden_new, shortcut_bonus = self._build_shortcut_context(hidden_new, nodes, q_rel, mode)
        
        # forward without node sampling
        if self.n_node_topk <= 0:
            return hidden_new

        # forward with node sampling
        # indexing sampling operation
        tmp_diff_node_idx = torch.ones(n_node, device=hidden_new.device)
        tmp_diff_node_idx[old_nodes_new_idx] = 0
        bool_diff_node_idx = tmp_diff_node_idx.bool().to(hidden_new.device)
        diff_node = nodes[bool_diff_node_idx]

        # project logit to fixed-size tensor via indexing
        diff_node_logit  = self.W_samp(hidden_new[bool_diff_node_idx]).squeeze(-1) # [all_batch_new_nodes]
        if shortcut_bonus is not None:
            diff_node_logit = diff_node_logit + self.shortcut_prune_lambda * shortcut_bonus[bool_diff_node_idx]
        
        # save logit to node_scores for later indexing
        node_scores = torch.ones((batchsize, self.n_ent), device=hidden_new.device) * float('-inf')
        node_scores[diff_node[:,0], diff_node[:,1]] = diff_node_logit

        # select top-k nodes
        # (train mode) self.softmax == F.gumbel_softmax
        # (eval mode)  self.softmax == F.softmax 
        node_scores = self.softmax(node_scores) # [batchsize, n_ent]
        topk_index = torch.topk(node_scores, self.n_node_topk, dim=1).indices.reshape(-1)
        topk_batchidx  = torch.arange(batchsize, device=hidden_new.device).repeat(self.n_node_topk,1).T.reshape(-1)
        batch_topk_nodes = torch.zeros((batchsize, self.n_ent), device=hidden_new.device)
        batch_topk_nodes[topk_batchidx, topk_index] = 1

        # get sampled nodes' relative index
        bool_sampled_diff_nodes_idx = batch_topk_nodes[diff_node[:,0], diff_node[:,1]].bool()
        bool_same_node_idx = ~bool_diff_node_idx
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
        self.attn_dim    = params.attn_dim
        self.n_ent       = params.n_ent
        self.n_rel       = params.n_rel
        self.n_node_topk = params.n_node_topk
        self.n_edge_topk = params.n_edge_topk
        self.loader      = loader
        self.shortcut_hops = getattr(params, 'shortcut_hops', 1)
        self.shortcut_topk = getattr(params, 'shortcut_topk', 0)
        self.shortcut_decay = getattr(params, 'shortcut_decay', 0.5)
        self.shortcut_lambda = getattr(params, 'shortcut_lambda', 0.0)
        self.shortcut_candidate_cap = getattr(params, 'shortcut_candidate_cap', 64)
        self.d_hop = getattr(params, 'd_hop', self.hidden_dim)
        self.shortcut_prune_lambda = getattr(params, 'shortcut_prune_lambda', -1.0)
        acts = {'relu': nn.ReLU(), 'tanh': torch.tanh, 'idd': lambda x:x}
        act  = acts[params.act]

        self.gnn_layers = []
        for i in range(self.n_layer):
            i_n_node_topk = self.n_node_topk if 'int' in str(type(self.n_node_topk)) else self.n_node_topk[i]
            self.gnn_layers.append(GNNLayer(self.hidden_dim, self.hidden_dim, self.attn_dim, self.n_rel, self.n_ent, \
                                            n_node_topk=i_n_node_topk, n_edge_topk=self.n_edge_topk, tau=params.tau, act=act, \
                                            loader=self.loader, shortcut_hops=self.shortcut_hops, shortcut_topk=self.shortcut_topk, \
                                            shortcut_decay=self.shortcut_decay, shortcut_lambda=self.shortcut_lambda, \
                                            shortcut_candidate_cap=self.shortcut_candidate_cap, d_hop=self.d_hop, \
                                            shortcut_prune_lambda=self.shortcut_prune_lambda))

        self.gnn_layers = nn.ModuleList(self.gnn_layers)       
        self.dropout = nn.Dropout(params.dropout)
        self.W_final = nn.Linear(self.hidden_dim, 1, bias=False)
        self.gate    = nn.GRU(self.hidden_dim, self.hidden_dim)

    def updateTopkNums(self, topk_list):
        assert len(topk_list) == self.n_layer
        for idx in range(self.n_layer):
            self.gnn_layers[idx].n_node_topk = topk_list[idx]

    def fixSamplingWeight(self):
        def freeze(m):
            m.requires_grad=False
        for i in range(self.n_layer):
            self.gnn_layers[i].W_samp.apply(freeze)

    def forward(self, subs, rels, mode='train'):
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

            # GNN forward -> get hidden representation at i-th layer
            # hidden: [k1, dim]
            hidden, nodes, sampled_nodes_idx = self.gnn_layers[i](q_sub, q_rel, hidden, edges, nodes, old_nodes_new_idx, n, mode=mode)

            # combine h0 and hi -> update hi with gate operation
            h0          = torch.zeros(1, n_node, hidden.size(1)).cuda().index_copy_(1, old_nodes_new_idx, h0)
            h0          = h0[0, sampled_nodes_idx, :].unsqueeze(0)
            hidden      = self.dropout(hidden)
            hidden, h0  = self.gate(hidden.unsqueeze(0), h0)
            hidden      = hidden.squeeze(0)
            
        # readout
        # [K, 2] (batch_idx, node_idx) K is w.r.t. n_nodes
        scores     = self.W_final(hidden).squeeze(-1)                   
        # non-visited entities have 0 scores
        scores_all = torch.zeros((n, self.loader.n_ent)).cuda()         
        # [B, n_all_nodes]
        scores_all[[nodes[:,0], nodes[:,1]]] = scores                   
        
        return scores_all
