"""
Phase 3: HDN-DDI architecture, reimplemented from Sun & Zheng (2025), BMC Bioinformatics,
Section "Method" (Eq. 1-12), operating on the hierarchical graphs built in Phase 2.

Since the authors' public code does not implement the paper's hierarchical substructure
module (see project notes), this is built directly from the paper's written equations,
cross-checked against their released code only where the two describe the same thing
(e.g. atom features, co-attention style).

Note on the interactive-view layer: a later revision tried to be more literally faithful
to Eq. 4-7 (explicit self-loop term, weight sharing with Eq. 7's plain transform, extra
sigmoid activations) via a hand-written attention layer. That version -- and a further
revision matching the authors' exact training hyperparameters (weight decay, LR decay,
200-epoch early stopping) -- both measured WORSE than this simpler version. This file is
deliberately reverted back to the configuration that produced our best verified result
(89.40% test accuracy / 95.23% AUROC on DrugBank warm-start). See project notes for the
full comparison across attempts.

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
from torch import nn
from torch_geometric.nn import GATConv


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

    Gap 1 (edge-aware attention): when use_edge_features=True, the hierarchical-view GAT
    incorporates real bond chemistry (bond type, conjugation, ring membership -- see
    molecular_graph.py's bond_features()) directly into the attention score, via PyG's
    native edge_dim support: e_ij = a^T[Wh_i || Wh_j || W_e*edge_attr_ij]. This is the
    principled formula, distinct from the naive constant-scalar edge gates that SSI-DDI,
    DSN-DDI, and HDN-DDI all report degrade performance -- see project notes for why this
    project targets that specific gap. Only the hierarchical-view layer uses this: the
    interactive-view layer connects substructures across two different drugs, which have
    no real bond between them, so edge features don't apply there."""

    def __init__(self, in_dim, hidden_dim=64, block_out_dim=128, heads=2, edge_drop_rate=0.1,
                 use_edge_features=False, edge_dim=6):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.edge_drop_rate = edge_drop_rate
        self.use_edge_features = use_edge_features

        # Eq. 1-3: hierarchical-view GAT, operates on the WHOLE hierarchical graph
        # (atom+substructure+molecule nodes together), independently for each drug.
        self.hierarchical_gat = GATConv(
            in_dim, hidden_dim, heads=heads, concat=False,
            edge_dim=edge_dim if use_edge_features else None,
        )

        # Eq. 4-6: interactive-view GAT, bipartite between the two drugs' substructure nodes.
        self.interactive_gat = GATConv((in_dim, in_dim), hidden_dim, heads=heads, concat=False, add_self_loops=False)

        # Eq. 7: plain nonlinear transform applied to non-substructure nodes.
        self.interactive_transform = nn.Linear(in_dim, hidden_dim)

        # Eq. 8: update layer - combine hierarchical-view + interactive-view representations,
        # output at block_out_dim (128), matching the paper's stated spec.
        self.update_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, block_out_dim),
            nn.ELU(),
        )

    def forward(self, batch_x, batch_y):
        x_x, x_y = batch_x.x, batch_y.x

        # --- hierarchical-view (Eq 1-3) ---
        if self.use_edge_features:
            h_x = self.hierarchical_gat(x_x, batch_x.edge_index, edge_attr=batch_x.edge_attr)
            h_y = self.hierarchical_gat(x_y, batch_y.edge_index, edge_attr=batch_y.edge_attr)
        else:
            h_x = self.hierarchical_gat(x_x, batch_x.edge_index)
            h_y = self.hierarchical_gat(x_y, batch_y.edge_index)

        # --- interactive-view (Eq 4-6), whole batch at once ---
        edge_xy = build_batched_bipartite_edges(
            batch_x.node_type, batch_x.batch, batch_y.node_type, batch_y.batch,
            drop_rate=self.edge_drop_rate, training=self.training,
        )
        inter_x = self.interactive_transform(x_x)  # Eq 7 default for non-substructure nodes
        inter_y = self.interactive_transform(x_y)
        if edge_xy.shape[1] > 0:
            size_xy = (x_x.size(0), x_y.size(0))
            y_from_x = self.interactive_gat((x_x, x_y), edge_xy, size=size_xy)
            x_from_y = self.interactive_gat((x_y, x_x), edge_xy.flip(0), size=size_xy[::-1])
            sub_mask_x = batch_x.node_type == 1
            sub_mask_y = batch_y.node_type == 1
            inter_x = torch.where(sub_mask_x.unsqueeze(-1), x_from_y, inter_x)
            inter_y = torch.where(sub_mask_y.unsqueeze(-1), y_from_x, inter_y)

        # --- update layer (Eq 8) ---
        new_x = self.update_mlp(torch.cat([h_x, inter_x], dim=-1))
        new_y = self.update_mlp(torch.cat([h_y, inter_y], dim=-1))
        return new_x, new_y


class HDNEncoder(nn.Module):
    """Stacks L HDN Blocks; collects the molecule-level node's representation from
    every block as that block's "global embedding" (Eq. 9), for every pair in the batch."""

    def __init__(self, in_dim, hidden_dim=64, block_out_dim=128, n_blocks=6, heads=2, edge_drop_rate=0.1,
                 use_edge_features=False, edge_dim=6):
        super().__init__()
        self.n_blocks = n_blocks
        self.blocks = nn.ModuleList()
        dim = in_dim
        for _ in range(n_blocks):
            self.blocks.append(HDNBlock(dim, hidden_dim, block_out_dim, heads=heads, edge_drop_rate=edge_drop_rate,
                                         use_edge_features=use_edge_features, edge_dim=edge_dim))
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
    def __init__(self, in_dim=55, hidden_dim=64, block_out_dim=128, n_blocks=6, heads=2, n_rel_types=86, edge_drop_rate=0.1,
                 use_edge_features=False, edge_dim=6):
        super().__init__()
        self.encoder = HDNEncoder(in_dim, hidden_dim, block_out_dim, n_blocks=n_blocks, heads=heads, edge_drop_rate=edge_drop_rate,
                                   use_edge_features=use_edge_features, edge_dim=edge_dim)
        self.decoder = HDNDecoder(block_out_dim, n_rel_types)

    def forward(self, batch_x, batch_y, rel_ids):
        g_x, g_y = self.encoder(batch_x, batch_y)
        return self.decoder(g_x, g_y, rel_ids)
