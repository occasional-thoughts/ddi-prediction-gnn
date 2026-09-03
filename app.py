"""
Streamlit demo UI for the Bond-Aware Hierarchical GNN DDI predictor.

Run with:
    streamlit run app.py

Give it two drugs (DrugBank ID or raw SMILES) and it predicts the interaction,
computed live by the trained model -- the same logic as src/demo_predict.py,
wrapped in a browser UI for live demonstration.
"""
import sys
import time

import altair as alt
import pandas as pd
import streamlit as st
import torch
from torch_geometric.data import Batch

sys.path.insert(0, "src")
from molecular_graph import smiles_to_hierarchical_graph
from model import HDN_DDI

DEVICE = torch.device("cpu")
N_REL_TYPES = 86

# (checkpoint path, use_edge_features, short accuracy note)
MODEL_OPTIONS = {
    "Warm-start, bond-aware (Gap 1) — 89.80% ACC [recommended]":
        ("results/hdn_ddi_warmstart_fold0_edgeaware.pt", True, "89.80% ACC on DrugBank warm-start"),
    "Warm-start, baseline / no bond features — 89.40% ACC":
        ("results/hdn_ddi_warmstart_fold0_v2.pt", False, "89.40% ACC on DrugBank warm-start"),
    "Cold-start fold0, baseline — tests generalization to unseen drugs":
        ("results/hdn_ddi_coldstart_fold0.pt", False, "58.30%/70.92% ACC on cold-start S1/S2"),
    "Cold-start fold1, baseline — tests generalization to unseen drugs":
        ("results/hdn_ddi_coldstart_fold1.pt", False, "62.34%/74.00% ACC on cold-start S1/S2"),
    "Cold-start fold2, baseline — tests generalization to unseen drugs":
        ("results/hdn_ddi_coldstart_fold2.pt", False, "58.21%/70.53% ACC on cold-start S1/S2"),
}

EXAMPLES = {
    "Example pair 1 (DrugBank IDs)": ("DB00460", "DB04571"),
    "Example pair 2 (Aspirin + Ibuprofen, raw SMILES)": (
        "CC(=O)OC1=CC=CC=C1C(=O)O", "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",
    ),
}

NAVY = "#1E2761"
ACCENT = "#C00000"

st.set_page_config(page_title="DDI Predictor", page_icon="🧬", layout="centered")

st.markdown(f"""
<style>
    .main .block-container {{ max-width: 780px; }}
    h1 {{ color: {NAVY}; }}
    .stButton>button {{ background-color: {NAVY}; color: white; font-weight: 600; border: none; }}
    .stButton>button:hover {{ background-color: {ACCENT}; color: white; }}
    .verdict-box {{ padding: 1rem 1.2rem; border-radius: 8px; border: 1.5px solid {NAVY};
                    background-color: #EEF1FA; margin-top: 1rem; }}
</style>
""", unsafe_allow_html=True)

st.title("🧬 Bond-Aware DDI Predictor")
st.caption("Bond-Aware Hierarchical GNN for Generalizable Drug–Drug Interaction Prediction")
st.write(
    "Enter two drugs as either a **DrugBank ID** (e.g. `DB00460`) or a **raw SMILES string**, "
    "and the trained model predicts the likelihood of an interaction, computed live."
)


@st.cache_resource(show_spinner="Loading trained model...")
def load_model(checkpoint_path, use_edge_features):
    model = HDN_DDI(in_dim=55, hidden_dim=64, n_blocks=6, heads=2, n_rel_types=N_REL_TYPES,
                     use_edge_features=use_edge_features).to(DEVICE)
    model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE, weights_only=True))
    model.eval()
    return model


@st.cache_resource(show_spinner=False)
def load_smiles_lookup():
    df = pd.read_csv("data/raw/DrugBank/drug_smiles.csv")
    return dict(zip(df["drug_id"], df["smiles"]))


@st.cache_resource(show_spinner=False)
def load_name_lookup():
    df = pd.read_csv("data/raw/DrugBank/drug_names.csv")
    return dict(zip(df["drugbank_id"], df["name"]))


@st.cache_resource(show_spinner=False)
def load_type_descriptions():
    df = pd.read_csv("data/raw/DrugBank/interaction_types.csv")
    return dict(zip(df["type_id"], df["description"]))


def resolve_drug(token, smiles_lookup, name_lookup):
    """Returns (smiles, id_or_input, display_label). display_label is 'Name (ID)' when
    the ID is a real DrugBank ID with a verified name, just the ID when it's a DrugBank
    ID we don't have a name for, or a truncated SMILES preview for raw SMILES input."""
    token = token.strip()
    if token in smiles_lookup:
        name = name_lookup.get(token)
        label = f"{name} ({token})" if name else f"{token} (name not in reference dataset)"
        return smiles_lookup[token], token, label
    preview = token if len(token) <= 24 else token[:21] + "..."
    return token, token, f"Custom SMILES: {preview}"


if "drug_a" not in st.session_state:
    st.session_state.drug_a, st.session_state.drug_b = EXAMPLES["Example pair 1 (DrugBank IDs)"]

st.subheader("Choose an example (optional)")
cols = st.columns(len(EXAMPLES))
for col, (label, (a, b)) in zip(cols, EXAMPLES.items()):
    if col.button(label, use_container_width=True):
        st.session_state.drug_a, st.session_state.drug_b = a, b

st.subheader("Or enter your own")
col1, col2 = st.columns(2)
drug_a_in = col1.text_input("Drug A (DrugBank ID or SMILES)", key="drug_a")
drug_b_in = col2.text_input("Drug B (DrugBank ID or SMILES)", key="drug_b")

model_choice = st.selectbox("Model / trained weights to use", list(MODEL_OPTIONS.keys()))
checkpoint_path, use_edge_features, acc_note = MODEL_OPTIONS[model_choice]
st.caption(f"Selected: {acc_note}. Warm-start weights are trained on drugs the model has "
           f"seen before; cold-start weights are separately trained and tested on drugs the "
           f"model has never seen, to measure generalization.")

predict = st.button("🔬 Predict Interaction", use_container_width=True, type="primary")

if predict:
    smiles_lookup = load_smiles_lookup()
    name_lookup = load_name_lookup()
    type_desc = load_type_descriptions()
    smiles_a, id_a, label_a = resolve_drug(drug_a_in, smiles_lookup, name_lookup)
    smiles_b, id_b, label_b = resolve_drug(drug_b_in, smiles_lookup, name_lookup)
    # for substituting into interaction sentences: prefer the verified name, fall back to
    # the ID, and leave the generic "Drug A"/"Drug B" wording alone for raw-SMILES input
    # we have no identity for at all
    sentence_a = name_lookup.get(id_a, id_a if id_a in smiles_lookup else "Drug A")
    sentence_b = name_lookup.get(id_b, id_b if id_b in smiles_lookup else "Drug B")

    def describe(type_id):
        text = type_desc.get(type_id, f"(no description found for type {type_id})")
        return text.replace("Drug A", sentence_a).replace("Drug B", sentence_b)

    graph_a = smiles_to_hierarchical_graph(smiles_a)
    graph_b = smiles_to_hierarchical_graph(smiles_b)

    if graph_a is None or graph_b is None:
        st.error("Could not parse one of the SMILES strings — check the input and try again.")
    else:
        model = load_model(checkpoint_path, use_edge_features)
        with st.spinner(f"Scoring all {N_REL_TYPES} interaction-type codes..."):
            start = time.time()
            scores = torch.zeros(N_REL_TYPES)
            with torch.no_grad():
                for rel in range(N_REL_TYPES):
                    bx = Batch.from_data_list([graph_a])
                    by = Batch.from_data_list([graph_b])
                    scores[rel] = model(bx, by, torch.tensor([rel])).item()
            elapsed = time.time() - start

        top5 = torch.topk(scores, 5)
        best_score = top5.values[0].item()
        best_type = top5.indices[0].item()

        st.markdown("---")
        st.subheader("Result")
        c1, c2 = st.columns(2)
        c1.metric("Drug A", label_a)
        c2.metric("Drug B", label_b)

        verdict = "⚠️ Interaction predicted" if best_score > 0.5 else "✅ No significant interaction predicted"
        st.markdown(
            f'<div class="verdict-box"><b>{verdict}</b><br>'
            f'{describe(best_type)}<br>'
            f'<span style="color:{NAVY};font-weight:600;">{best_score*100:.2f}% confidence '
            f'(type code {best_type})</span></div>', unsafe_allow_html=True,
        )

        def ranked_bar_chart(indices, values, show_axis_labels, height=280):
            df = pd.DataFrame({
                "rank": list(range(len(indices))),
                "type_label": [f"Type {i}" for i in indices],
                "description": [describe(i) for i in indices],
                "probability": [v * 100 for v in values],
            })
            x_enc = alt.X("rank:O", title=None, axis=alt.Axis(labels=show_axis_labels, ticks=show_axis_labels))
            chart = alt.Chart(df).mark_bar(color=NAVY).encode(
                x=x_enc,
                y=alt.Y("probability:Q", title="Probability (%)", scale=alt.Scale(domain=[0, 100])),
                tooltip=["type_label", "description", alt.Tooltip("probability:Q", format=".2f")],
            ).properties(height=height)
            st.altair_chart(chart, use_container_width=True)

        st.write("")
        st.write("**Top 5 predicted interaction types** (hover a bar for the full sentence):")
        ranked_bar_chart(top5.indices.tolist(), top5.values.tolist(), show_axis_labels=True)
        for i, v in zip(top5.indices.tolist(), top5.values.tolist()):
            st.markdown(f"- **{v*100:.2f}%** — {describe(i)} *(type {i})*")

        st.caption(
            "ℹ️ Descriptions come from the original DeepDDI dataset release (Ryu et al. 2018, PNAS), "
            "the source of DrugBank's standard 86-type DDI benchmark that this project's dataset "
            "also uses. That file's own type numbering didn't line up directly with this dataset's "
            "0-85 codes, so the mapping was cross-checked by matching frequency patterns — e.g. "
            "this project's single most common type (31.7% of all pairs) lines up with DeepDDI's "
            "generic \"risk of adverse effects increased\" category, the expected dominant class in "
            "this kind of benchmark — rather than assumed outright."
        )

        with st.expander(f"Show all {N_REL_TYPES} interaction-type scores"):
            all_sorted = torch.argsort(scores, descending=True)
            ranked_bar_chart(all_sorted.tolist(), scores[all_sorted].tolist(), show_axis_labels=False, height=280)
            st.dataframe(
                pd.DataFrame({
                    "Type code": all_sorted.tolist(),
                    "Description": [describe(i) for i in all_sorted.tolist()],
                    "Probability (%)": [f"{scores[i].item()*100:.2f}" for i in all_sorted],
                }),
                hide_index=True, use_container_width=True,
            )

        st.caption(f"Computed in {elapsed:.2f}s on CPU, using: {model_choice} ({acc_note}).")

st.markdown("---")
st.caption(
    "A Antony Abisha · R Sriharini · Pranav S Easwar — BCSE497J Project-I, "
    "under the supervision of Dheeba J., School of Computer Science and Engineering (SCOPE)"
)
