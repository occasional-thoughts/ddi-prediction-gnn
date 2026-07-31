"""
Phase 3: HDN-DDI architecture, reimplemented from Sun & Zheng (2025), BMC Bioinformatics,
Section "Method" (Eq. 1-12), operating on the hierarchical graphs built in Phase 2.

Since the authors' public code does not implement the paper's hierarchical substructure
module (see project notes), this is built directly from the paper's written equations,
cross-checked against their released code only where the two describe the same thing
(e.g. atom features, co-attention style).

Node levels (set in Phase 2's molecular_graph.py):
  node_type == 0  ->  atom-level
  node_type == 1  ->  substructure-level  (BRICS fragments)
  node_type == 2  ->  molecule-level      (single node per drug)

Everything here operates on BATCHES of drug pairs at once (many pairs processed in one
vectorized forward pass), not one pair at a time -- this is what makes training on
~190K+ DrugBank pairs actually feasible; see project notes on the ~10ms/pair unbatched
timing that made a full epoch take over an hour.
"""
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GATConv
from torch_geometric.utils import softmax as pyg_softmax


class SharedInteractiveGAT(nn.Module):
    """Interactive-view GAT (Eq. 4-6), reimplemented by hand instead of using PyG's
    GATConv, because two things the paper requires can't be expressed with it:

    1. Eq. 4 aggregates over N-tilde_i UNION {i} -- each node must include its own
       previous features in its own update, not just messages from the other drug.
       PyG's bipartite GATConv has no self-loop concept when src and dst are two
       different tensors (self-loops only mean something when src IS dst).
    2. Eq. 7's plain transform explicitly "shares the same weight with the GAT in
       the interactive-view layer." PyG's bipartite GATConv keeps separate lin_src/
       lin_dst weights internally and doesn't expose them for reuse elsewhere.

    Both are solved by holding ONE nn.Linear (self.W) and reusing it everywhere:
    for the attention-key projection, the self-term, and Eq. 7's plain_transform.
    """

    def __init__(self, in_dim, out_dim, heads=2, negative_slope=0.2):
        super().__init__()
        self.heads = heads
        self.out_dim = out_dim
        self.W = nn.Linear(in_dim, heads * out_dim, bias=False)  # shared W-tilde^(l+1)
        self.bias = nn.Parameter(torch.zeros(out_dim))  # shared b-tilde^(l+1)
        self.att = nn.Parameter(torch.empty(1, heads, 2 * out_dim))  # a-tilde^(l+1)
        self.negative_slope = negative_slope
        nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_uniform_(self.att)

    def plain_transform(self, x):
        """Eq. 7: sigma(W-tilde x + b-tilde), same W/b as the attention layer below.
        No extra activation applied here (see note in forward()) -- Eq.8's ELU already
        provides the per-block nonlinearity."""
        Wx = self.W(x).view(x.size(0), self.heads, self.out_dim).mean(dim=1)
        return Wx + self.bias

    def forward(self, x_src, x_dst, edge_index):
        """edge_index: [2, E] cross-drug edges only (row0 indexes x_src, row1 indexes
        x_dst). The 'U {i}' self-term is added internally, not passed in."""
        n_src, n_dst = x_src.size(0), x_dst.size(0)
        Wh_src = self.W(x_src).view(n_src, self.heads, self.out_dim)
        Wh_dst = self.W(x_dst).view(n_dst, self.heads, self.out_dim)

        self_idx = torch.arange(n_dst, device=x_dst.device)
        src_idx = torch.cat([edge_index[0], self_idx + n_src])  # self-loops point into the appended Wh_dst block
        dst_idx = torch.cat([edge_index[1], self_idx])
        Wh_pool = torch.cat([Wh_src, Wh_dst], dim=0)

        e = (torch.cat([Wh_pool[src_idx], Wh_dst[dst_idx]], dim=-1) * self.att).sum(dim=-1)  # [E', heads]
        e = F.leaky_relu(e, self.negative_slope)
        alpha = pyg_softmax(e, dst_idx, num_nodes=n_dst)  # per-head softmax, grouped by destination

        out = x_dst.new_zeros(n_dst, self.heads, self.out_dim)
        out.index_add_(0, dst_idx, Wh_pool[src_idx] * alpha.unsqueeze(-1))
        out = out.mean(dim=1) + self.bias
        return out


def build_batched_bipartite_edges(node_type_x, batch_idx_x, node_type_y, batch_idx_y, drop_rate=0.0, training=True):
    """Interactive-view bipartite graph (Eq. 4-6) for an entire batch of drug pairs at once.
    Connects every substructure-level node of pair i's drug X to every substructure-level
    node of pair i's drug Y, for every pair i in the batch -- never across different pairs.
    Also randomly drops a fraction of edges during training (the paper's stated fix for
    redundant/adjacent substructures dominating attention -- see GMPNN-CS's unsolved version
    of this same problem in the project notes).
    Returns a [2, E] edge_index indexed into the FULL (atom+substructure+molecule) node
    numbering of batch_x / batch_y -- GATConv only touches the nodes actually referenced."""
    sub_idx_x = (node_type_x == 1).nonzero(as_tuple=True)[0]
    sub_idx_y = (node_type_y == 1).nonzero(as_tuple=True)[0]
    sub_batch_x = batch_idx_x[sub_idx_x]
    sub_batch_y = batch_idx_y[sub_idx_y]

    batch_size = int(max(batch_idx_x.max().item(), batch_idx_y.max().item())) + 1
    src_list, dst_list = [], []
    for b in range(batch_size):
        xs = sub_idx_x[sub_batch_x == b]
        ys = sub_idx_y[sub_batch_y == b]
        if len(xs) == 0 or len(ys) == 0:
            continue
        src_list.append(xs.repeat_interleave(len(ys)))
        dst_list.append(ys.repeat(len(xs)))

    if not src_list:
        return torch.zeros((2, 0), dtype=torch.long, device=node_type_x.device)

    src, dst = torch.cat(src_list), torch.cat(dst_list)
    if training and drop_rate > 0:
        keep = torch.rand(len(src), device=src.device) > drop_rate
        src, dst = src[keep], dst[keep]
    return torch.stack([src, dst])


class HDNBlock(nn.Module):
    """One HDN Block = hierarchical-view layer + interactive-view layer + update layer
    (paper Section 'HDN encoder', Eq. 1-9), batched across many drug pairs at once.

    Per the paper's "Parameters" section: hierarchical-view and interactive-view layers
    each produce a 64-dim representation, but the UPDATE layer generates a 128-dim
    representation (not compressed back down to 64) -- that 128-dim output is what
    carries forward into the next block and becomes each block's global embedding."""

    def __init__(self, in_dim, hidden_dim=64, block_out_dim=128, heads=2, edge_drop_rate=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.edge_drop_rate = edge_drop_rate

        # Eq. 1-3: hierarchical-view GAT, operates on the WHOLE hierarchical graph
        # (atom+substructure+molecule nodes together), independently for each drug.
        # GATConv already includes self-loops by default, matching Eq.1's "U {i}" term.
        self.hierarchical_gat = GATConv(in_dim, hidden_dim, heads=heads, concat=False)

        # Eq. 4-7: interactive-view GAT + Eq.7's shared-weight plain transform,
        # both handled by one module so the weight sharing the paper requires is real.
        self.interactive = SharedInteractiveGAT(in_dim, hidden_dim, heads=heads)

        # Eq. 8: update layer - combine hierarchical-view + interactive-view representations,
        # output at block_out_dim (128), matching the paper's stated spec.
        self.update_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, block_out_dim),
            nn.ELU(),
        )

    def forward(self, batch_x, batch_y):
        x_x, x_y = batch_x.x, batch_y.x

        # --- hierarchical-view (Eq 1-3), sigma = sigmoid per the paper's activation note ---
        h_x = torch.sigmoid(self.hierarchical_gat(x_x, batch_x.edge_index))
        h_y = torch.sigmoid(self.hierarchical_gat(x_y, batch_y.edge_index))

        # --- interactive-view (Eq 4-7), whole batch at once ---
        edge_xy = build_batched_bipartite_edges(
            batch_x.node_type, batch_x.batch, batch_y.node_type, batch_y.batch,
            drop_rate=self.edge_drop_rate, training=self.training,
        )
        inter_x = self.interactive.plain_transform(x_x)  # Eq 7 default for non-substructure nodes
        inter_y = self.interactive.plain_transform(x_y)
        if edge_xy.shape[1] > 0:
            sub_mask_x = batch_x.node_type == 1
            sub_mask_y = batch_y.node_type == 1
            sub_idx_x = sub_mask_x.nonzero(as_tuple=True)[0]
            sub_idx_y = sub_mask_y.nonzero(as_tuple=True)[0]

            remap_x = x_x.new_full((x_x.size(0),), -1, dtype=torch.long)
            remap_x[sub_idx_x] = torch.arange(len(sub_idx_x), device=x_x.device)
            remap_y = x_y.new_full((x_y.size(0),), -1, dtype=torch.long)
            remap_y[sub_idx_y] = torch.arange(len(sub_idx_y), device=x_y.device)
            local_edge = torch.stack([remap_x[edge_xy[0]], remap_y[edge_xy[1]]])

            y_update = self.interactive(x_x[sub_idx_x], x_y[sub_idx_y], local_edge)
            x_update = self.interactive(x_y[sub_idx_y], x_x[sub_idx_x], local_edge.flip(0))

            inter_x = inter_x.clone()
            inter_y = inter_y.clone()
            inter_x[sub_idx_x] = x_update
            inter_y[sub_idx_y] = y_update

        # --- update layer (Eq 8) ---
        new_x = self.update_mlp(torch.cat([h_x, inter_x], dim=-1))
        new_y = self.update_mlp(torch.cat([h_y, inter_y], dim=-1))
        return new_x, new_y


class HDNEncoder(nn.Module):
    """Stacks L HDN Blocks; collects the molecule-level node's representation from
    every block as that block's "global embedding" (Eq. 9), for every pair in the batch."""

    def __init__(self, in_dim, hidden_dim=64, block_out_dim=128, n_blocks=6, heads=2, edge_drop_rate=0.1):
        super().__init__()
        self.n_blocks = n_blocks
        self.blocks = nn.ModuleList()
        dim = in_dim
        for _ in range(n_blocks):
            self.blocks.append(HDNBlock(dim, hidden_dim, block_out_dim, heads=heads, edge_drop_rate=edge_drop_rate))
            dim = block_out_dim  # every block after the first receives the 128-dim update output

    def forward(self, batch_x, batch_y):
        global_embs_x, global_embs_y = [], []
        for block in self.blocks:
            new_x, new_y = block(batch_x, batch_y)
            batch_x.x, batch_y.x = new_x, new_y
            global_embs_x.append(new_x[batch_x.mol_node_idx])  # [batch_size, block_out_dim]
            global_embs_y.append(new_y[batch_y.mol_node_idx])
        # [n_blocks, batch_size, block_out_dim] -> [batch_size, n_blocks, block_out_dim]
        return torch.stack(global_embs_x, dim=1), torch.stack(global_embs_y, dim=1)


class HDNDecoder(nn.Module):
    """Co-attention across block-depths (Eq. 10) + relation-aware bilinear scoring (Eq. 11)."""

    def __init__(self, hidden_dim, n_rel_types):
        super().__init__()
        self.w_x = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.w_y = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.alpha = nn.Parameter(torch.randn(hidden_dim))
        self.rel_matrix = nn.Embedding(n_rel_types, hidden_dim * hidden_dim)
        nn.init.xavier_uniform_(self.rel_matrix.weight)
        self.hidden_dim = hidden_dim

    def forward(self, g_x, g_y, rel_ids):
        # g_x, g_y: [batch, n_blocks, hidden_dim]
        # Eq. 10: gamma_{l,l'} = alpha^T tanh(W_x g_x^(l) + W_y g_y^(l'))
        proj_x = self.w_x(g_x).unsqueeze(2)  # [batch, L, 1, hidden]
        proj_y = self.w_y(g_y).unsqueeze(1)  # [batch, 1, L, hidden]
        gamma = torch.tanh(proj_x + proj_y) @ self.alpha  # [batch, L, L]

        m_r = self.rel_matrix(rel_ids).view(-1, self.hidden_dim, self.hidden_dim)  # [batch, hidden, hidden]

        # Eq. 11: s = sigmoid( sum_l sum_l' gamma_{l,l'} * (g_x^(l))^T M_r g_y^(l') )
        gx_Mr = torch.einsum('blh,bhk->blk', g_x, m_r)      # [batch, L, hidden]
        pairwise = torch.einsum('blk,bmk->blm', gx_Mr, g_y)  # [batch, L, L]
        score = (gamma * pairwise).sum(dim=(1, 2))           # [batch]
        return torch.sigmoid(score)


class HDN_DDI(nn.Module):
    def __init__(self, in_dim=55, hidden_dim=64, block_out_dim=128, n_blocks=6, heads=2, n_rel_types=86, edge_drop_rate=0.1):
        super().__init__()
        self.encoder = HDNEncoder(in_dim, hidden_dim, block_out_dim, n_blocks=n_blocks, heads=heads, edge_drop_rate=edge_drop_rate)
        self.decoder = HDNDecoder(block_out_dim, n_rel_types)

    def forward(self, batch_x, batch_y, rel_ids):
        g_x, g_y = self.encoder(batch_x, batch_y)
        return self.decoder(g_x, g_y, rel_ids)
