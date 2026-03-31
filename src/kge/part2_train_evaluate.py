"""
TD5 - Part 2, Steps 2-8: KGE Training, Evaluation & Analysis
==============================================================
Train TransE + ComplEx with PyKEEN, evaluate link prediction,
run KB-size sensitivity, embedding analysis, and rule comparison.
"""
import os
import json
import warnings
import numpy as np

warnings.filterwarnings("ignore")

BASE    = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(BASE, "data")
RES_DIR = os.path.join(OUT_DIR, "results")

# ── Hyper-parameters (consistent across models for fair comparison) ──
CONFIG = {
    "embedding_dim":  100,
    "learning_rate":  0.01,
    "batch_size":     512,
    "num_epochs":     50,
    "neg_sampler":    "basic",            # basic negative sampling
    "neg_per_pos":    1,
    "evaluator":      "RankBasedEvaluator",
    "filtered":       True,
}

# Key entities for nearest-neighbor analysis
KEY_ENTITIES = [
    "http://dbpedia.org/resource/Mario",
    "http://dbpedia.org/resource/Nintendo",
    "http://dbpedia.org/resource/Shigeru_Miyamoto",
    "http://dbpedia.org/resource/Super_Mario_Bros.",
    "http://dbpedia.org/resource/Bowser_(character)",
    "http://dbpedia.org/resource/Game_Boy",
    "http://dbpedia.org/resource/Luigi",
    "http://dbpedia.org/resource/The_Legend_of_Zelda",
    "http://esilv.fr/mario#Mario",
    "http://esilv.fr/mario#Luigi",
]


# ─────────────────────────────────────────────────────────────
# Section 2-3: Model Training
# ─────────────────────────────────────────────────────────────
def train_model(model_name, training_tf, testing_tf, validation_tf,
                epochs=None, device="cpu"):
    """Train a single PyKEEN model and return the pipeline result."""
    from pykeen.pipeline import pipeline

    n_epochs = epochs or CONFIG["num_epochs"]
    print(f"\n  Training {model_name} (dim={CONFIG['embedding_dim']}, "
          f"lr={CONFIG['learning_rate']}, epochs={n_epochs}) ...")

    result = pipeline(
        training=training_tf,
        testing=testing_tf,
        validation=validation_tf,
        model=model_name,
        model_kwargs=dict(embedding_dim=CONFIG["embedding_dim"]),
        optimizer="Adam",
        optimizer_kwargs=dict(lr=CONFIG["learning_rate"]),
        training_kwargs=dict(
            num_epochs=n_epochs,
            batch_size=CONFIG["batch_size"],
        ),
        negative_sampler=CONFIG["neg_sampler"],
        negative_sampler_kwargs=dict(num_negs_per_pos=CONFIG["neg_per_pos"]),
        evaluator_kwargs=dict(filtered=CONFIG["filtered"]),
        device=device,
        random_seed=42,
    )
    return result


def extract_metrics(result):
    """Extract MRR, Hits@1/3/10 from a PyKEEN result."""
    metrics = result.metric_results.to_dict()
    # PyKEEN stores metrics in a nested dict; extract key ones
    out = {}
    for side in ["head", "tail", "both"]:
        prefix = side
        try:
            out[f"{prefix}_mrr"]     = metrics.get("both", metrics).get(
                "realistic", {}
            ).get("inverse_harmonic_mean_rank", None)
            out[f"{prefix}_hits@1"]  = metrics.get("both", metrics).get(
                "realistic", {}
            ).get("hits_at_1", None)
            out[f"{prefix}_hits@3"]  = metrics.get("both", metrics).get(
                "realistic", {}
            ).get("hits_at_3", None)
            out[f"{prefix}_hits@10"] = metrics.get("both", metrics).get(
                "realistic", {}
            ).get("hits_at_10", None)
        except Exception:
            pass

    # Simpler: use the flat metrics
    flat = {}
    try:
        flat["mrr"]      = result.metric_results.get_metric("both.realistic.inverse_harmonic_mean_rank")
        flat["hits@1"]   = result.metric_results.get_metric("both.realistic.hits_at_1")
        flat["hits@3"]   = result.metric_results.get_metric("both.realistic.hits_at_3")
        flat["hits@10"]  = result.metric_results.get_metric("both.realistic.hits_at_10")
    except Exception:
        # Fallback: try to parse from dict
        try:
            m = result.metric_results.to_flat_dict()
            flat["mrr"]    = m.get("both.realistic.inverse_harmonic_mean_rank", 0)
            flat["hits@1"] = m.get("both.realistic.hits_at_1", 0)
            flat["hits@3"] = m.get("both.realistic.hits_at_3", 0)
            flat["hits@10"]= m.get("both.realistic.hits_at_10", 0)
        except Exception:
            flat = {"mrr": 0, "hits@1": 0, "hits@3": 0, "hits@10": 0}

    return flat


def train_both_models(device="cpu"):
    """
    Section 2-3: Train TransE and ComplEx on the prepared splits.
    Returns dict with both results.
    """
    from pykeen.triples import TriplesFactory

    print("\n" + "=" * 60)
    print("Section 2-3: Training Embedding Models")
    print("=" * 60)

    # Load splits
    train_path = os.path.join(OUT_DIR, "train.txt")
    valid_path = os.path.join(OUT_DIR, "valid.txt")
    test_path  = os.path.join(OUT_DIR, "test.txt")

    training   = TriplesFactory.from_path(train_path)
    # Use same entity/relation mapping for valid and test
    validation = TriplesFactory.from_path(
        valid_path,
        entity_to_id=training.entity_to_id,
        relation_to_id=training.relation_to_id,
    )
    testing = TriplesFactory.from_path(
        test_path,
        entity_to_id=training.entity_to_id,
        relation_to_id=training.relation_to_id,
    )

    print(f"  Training triples:   {training.num_triples}")
    print(f"  Validation triples: {validation.num_triples}")
    print(f"  Testing triples:    {testing.num_triples}")
    print(f"  Entities:           {training.num_entities}")
    print(f"  Relations:          {training.num_relations}")

    results = {}
    for name in ["TransE", "ComplEx"]:
        res = train_model(name, training, testing, validation, device=device)
        metrics = extract_metrics(res)
        results[name] = {"result": res, "metrics": metrics}
        print(f"\n  {name} Results:")
        for k, v in metrics.items():
            print(f"    {k:>10}: {v:.4f}" if isinstance(v, float) else f"    {k:>10}: {v}")

    # Save metrics
    os.makedirs(RES_DIR, exist_ok=True)
    comparison = {name: data["metrics"] for name, data in results.items()}
    with open(os.path.join(RES_DIR, "model_comparison.json"), "w") as f:
        json.dump(comparison, f, indent=2)

    return results, training


# ─────────────────────────────────────────────────────────────
# Section 4: Evaluation (already done inside train_both_models)
# ─────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────
# Section 5.1: Model Comparison
# ─────────────────────────────────────────────────────────────
def compare_models(results):
    """Print model comparison table."""
    print("\n" + "=" * 60)
    print("Section 5.1: Model Comparison")
    print("=" * 60)
    header = f"  {'Model':<10} {'MRR':>8} {'H@1':>8} {'H@3':>8} {'H@10':>8}"
    print(header)
    print("  " + "-" * 46)
    for name, data in results.items():
        m = data["metrics"]
        print(f"  {name:<10} {m.get('mrr',0):>8.4f} "
              f"{m.get('hits@1',0):>8.4f} {m.get('hits@3',0):>8.4f} "
              f"{m.get('hits@10',0):>8.4f}")

    # Determine best model
    best = max(results, key=lambda n: results[n]["metrics"].get("mrr", 0))
    print(f"\n  Best model by MRR: {best}")
    return best


# ─────────────────────────────────────────────────────────────
# Section 5.2: KB Size Sensitivity
# ─────────────────────────────────────────────────────────────
def kb_size_experiment(best_model_name, device="cpu"):
    """Train best model on 20k and 50k subsampled triples."""
    from pykeen.triples import TriplesFactory

    print("\n" + "=" * 60)
    print("Section 5.2: KB Size Sensitivity")
    print("=" * 60)

    # Load full cleaned triples
    all_triples = []
    tsv_path = os.path.join(OUT_DIR, "cleaned_triples.tsv")
    with open(tsv_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) == 3:
                all_triples.append(tuple(parts))

    rng = np.random.RandomState(42)
    sizes = [20000, 50000, len(all_triples)]
    size_results = {}

    for size in sizes:
        label = f"{size//1000}k" if size < len(all_triples) else "full"
        print(f"\n  --- {label} triples ({size}) ---")

        if size < len(all_triples):
            idx = rng.choice(len(all_triples), size, replace=False)
            subset = [all_triples[i] for i in idx]
        else:
            subset = all_triples

        # Quick split
        arr = np.array(subset)
        n = len(arr)
        perm = rng.permutation(n)
        n_train = int(n * 0.8)
        n_valid = int(n * 0.1)

        tf_all = TriplesFactory.from_labeled_triples(arr[perm])
        training, rest = tf_all.split([0.8, 0.2], random_state=42)
        validation, testing = rest.split([0.5, 0.5], random_state=42)

        res = train_model(
            best_model_name, training, testing, validation,
            epochs=25,  # fewer epochs for speed
            device=device,
        )
        metrics = extract_metrics(res)
        size_results[label] = {
            "triples": size,
            "entities": training.num_entities,
            "relations": training.num_relations,
            **metrics,
        }
        print(f"    MRR={metrics.get('mrr',0):.4f}  "
              f"H@10={metrics.get('hits@10',0):.4f}")

    with open(os.path.join(RES_DIR, "size_experiment.json"), "w") as f:
        json.dump(size_results, f, indent=2)

    return size_results


# ─────────────────────────────────────────────────────────────
# Section 6.1: Nearest Neighbors
# ─────────────────────────────────────────────────────────────
def nearest_neighbors(result, training_tf, k=5):
    """Find k nearest neighbors for key entities."""
    print("\n" + "=" * 60)
    print("Section 6.1: Nearest Neighbors Analysis")
    print("=" * 60)

    import torch

    model = result.model
    model.eval()

    ent_emb = model.entity_representations[0](
        indices=None
    ).detach().cpu().numpy()

    # Handle complex embeddings (ComplEx)
    if np.iscomplexobj(ent_emb):
        ent_emb = np.real(ent_emb)

    e2id = training_tf.entity_to_id
    id2e  = {v: k for k, v in e2id.items()}

    # Normalize for cosine similarity
    norms = np.linalg.norm(ent_emb, axis=1, keepdims=True)
    norms[norms == 0] = 1
    ent_normed = ent_emb / norms

    nn_results = {}
    for uri in KEY_ENTITIES:
        if uri not in e2id:
            continue
        idx = e2id[uri]
        sims = ent_normed @ ent_normed[idx]
        top_idx = np.argsort(-sims)[1:k+1]  # skip self

        label = uri.rsplit("/", 1)[-1].rsplit("#", 1)[-1]
        neighbors = []
        print(f"\n  {label}:")
        for rank, ni in enumerate(top_idx, 1):
            n_uri = id2e[ni]
            n_label = n_uri.rsplit("/", 1)[-1].rsplit("#", 1)[-1]
            sim = sims[ni]
            print(f"    {rank}. {n_label} (sim={sim:.4f})")
            neighbors.append({"entity": n_uri, "label": n_label, "sim": float(sim)})
        nn_results[uri] = neighbors

    with open(os.path.join(RES_DIR, "nearest_neighbors.json"), "w") as f:
        json.dump(nn_results, f, indent=2)

    return nn_results


# ─────────────────────────────────────────────────────────────
# Section 6.2: Clustering / t-SNE Visualization
# ─────────────────────────────────────────────────────────────
def tsne_visualization(result, training_tf, max_entities=5000):
    """t-SNE 2D visualization of entity embeddings colored by type."""
    print("\n" + "=" * 60)
    print("Section 6.2: t-SNE Clustering Visualization")
    print("=" * 60)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE

    model = result.model
    model.eval()

    ent_emb = model.entity_representations[0](
        indices=None
    ).detach().cpu().numpy()

    # For ComplEx, embeddings may be complex; take real part or magnitude
    if ent_emb.dtype == np.complex64 or ent_emb.dtype == np.complex128:
        ent_emb = np.abs(ent_emb)

    e2id = training_tf.entity_to_id
    id2e = {v: k for k, v in e2id.items()}

    # Load entity type map
    type_map_path = os.path.join(OUT_DIR, "entity_types.json")
    if os.path.exists(type_map_path):
        with open(type_map_path) as f:
            type_map = json.load(f)
    else:
        type_map = {}

    n_ent = ent_emb.shape[0]
    if n_ent > max_entities:
        rng = np.random.RandomState(42)
        sample_idx = rng.choice(n_ent, max_entities, replace=False)
    else:
        sample_idx = np.arange(n_ent)

    emb_sample = ent_emb[sample_idx]

    print(f"  Running t-SNE on {len(sample_idx)} entities ...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
    coords = tsne.fit_transform(emb_sample)

    # Assign colors by ontology class
    TOP_TYPES = ["VideoGame", "Person", "Character", "Organization",
                 "Enemy", "Location", "Item", "Software", "Agent"]
    color_map = {t: i for i, t in enumerate(TOP_TYPES)}

    colors = []
    for idx in sample_idx:
        uri = id2e.get(idx, "")
        etype = type_map.get(uri, "Other")
        colors.append(color_map.get(etype, len(TOP_TYPES)))

    cmap = plt.cm.get_cmap("tab10", len(TOP_TYPES) + 1)

    fig, ax = plt.subplots(figsize=(12, 8))
    scatter = ax.scatter(
        coords[:, 0], coords[:, 1],
        c=colors, cmap=cmap, alpha=0.5, s=8,
    )

    # Legend
    handles = [
        plt.Line2D([0], [0], marker='o', color='w',
                   markerfacecolor=cmap(color_map[t]), markersize=8, label=t)
        for t in TOP_TYPES if color_map[t] in set(colors)
    ]
    handles.append(
        plt.Line2D([0], [0], marker='o', color='w',
                   markerfacecolor=cmap(len(TOP_TYPES)), markersize=8, label="Other")
    )
    ax.legend(handles=handles, loc="best", fontsize=8)
    ax.set_title("t-SNE of Entity Embeddings (colored by ontology class)")
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")

    plot_path = os.path.join(RES_DIR, "tsne_entities.png")
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved -> {plot_path}")

    return plot_path


# ─────────────────────────────────────────────────────────────
# Section 6.3: Relation Behavior Analysis
# ─────────────────────────────────────────────────────────────
def relation_behavior(result, training_tf):
    """Analyze symmetric, inverse, and composition relations."""
    print("\n" + "=" * 60)
    print("Section 6.3: Relation Behavior Analysis")
    print("=" * 60)

    model = result.model
    model.eval()

    rel_emb = model.relation_representations[0](
        indices=None
    ).detach().cpu().numpy()
    if rel_emb.dtype in (np.complex64, np.complex128):
        rel_emb = np.real(rel_emb)

    r2id = training_tf.relation_to_id
    id2r = {v: k for k, v in r2id.items()}

    norms = np.linalg.norm(rel_emb, axis=1, keepdims=True)
    norms[norms == 0] = 1
    rel_normed = rel_emb / norms

    # Check known semantic properties
    analysis = {}

    # 1. Symmetric relations: owl:sameAs, isSiblingOf
    #    For TransE, symmetric r should have ||r|| ≈ 0
    print("\n  Symmetric relations (expected small norm for TransE):")
    sym_candidates = [
        "http://www.w3.org/2002/07/owl#sameAs",
        "http://esilv.fr/mario#coOccursWith",
    ]
    sym_results = []
    for uri in sym_candidates:
        if uri not in r2id:
            continue
        idx = r2id[uri]
        norm = float(np.linalg.norm(rel_emb[idx]))
        label = uri.rsplit("/", 1)[-1].rsplit("#", 1)[-1]
        print(f"    {label}: ||r|| = {norm:.4f}")
        sym_results.append({"relation": uri, "label": label, "norm": norm})
    analysis["symmetric"] = sym_results

    # 2. Find most similar relation pairs (potential inverses)
    print("\n  Top similar relation pairs (potential inverses/synonyms):")
    sim_matrix = rel_normed @ rel_normed.T
    np.fill_diagonal(sim_matrix, -1)
    inv_results = []
    flat_idx = np.argsort(-sim_matrix.ravel())[:10]
    seen_pairs = set()
    for fi in flat_idx:
        i, j = divmod(fi, sim_matrix.shape[1])
        if (j, i) in seen_pairs:
            continue
        seen_pairs.add((i, j))
        r1 = id2r[i].rsplit("/", 1)[-1].rsplit("#", 1)[-1]
        r2 = id2r[j].rsplit("/", 1)[-1].rsplit("#", 1)[-1]
        sim = float(sim_matrix[i, j])
        print(f"    {r1} <-> {r2}: sim={sim:.4f}")
        inv_results.append({"r1": id2r[i], "r2": id2r[j], "sim": sim})
    analysis["similar_pairs"] = inv_results

    # 3. Composition check: r1 + r2 ≈ r3 ?
    #    Check: type + computingPlatform ≈ platforms ?
    print("\n  Composition check (r1 + r2 closest to r3):")
    comp_tests = [
        ("http://www.w3.org/1999/02/22-rdf-syntax-ns#type",
         "http://dbpedia.org/ontology/computingPlatform"),
    ]
    comp_results = []
    for r1_uri, r2_uri in comp_tests:
        if r1_uri not in r2id or r2_uri not in r2id:
            continue
        v = rel_emb[r2id[r1_uri]] + rel_emb[r2id[r2_uri]]
        v_norm = v / (np.linalg.norm(v) + 1e-8)
        sims = rel_normed @ v_norm
        top3 = np.argsort(-sims)[:3]
        r1_l = r1_uri.rsplit("/", 1)[-1].rsplit("#", 1)[-1]
        r2_l = r2_uri.rsplit("/", 1)[-1].rsplit("#", 1)[-1]
        print(f"    {r1_l} + {r2_l} closest to:")
        for rank, ti in enumerate(top3, 1):
            tl = id2r[ti].rsplit("/", 1)[-1].rsplit("#", 1)[-1]
            print(f"      {rank}. {tl} (sim={sims[ti]:.4f})")
            comp_results.append({
                "composition": f"{r1_l}+{r2_l}",
                "rank": rank,
                "nearest": tl,
                "sim": float(sims[ti]),
            })
    analysis["composition"] = comp_results

    with open(os.path.join(RES_DIR, "relation_analysis.json"), "w") as f:
        json.dump(analysis, f, indent=2)

    return analysis


# ─────────────────────────────────────────────────────────────
# Section 8: Rule vs Embedding Comparison
# ─────────────────────────────────────────────────────────────
def rule_vs_embedding(transe_result, training_tf):
    """
    Exercise 8: Design a SWRL rule for the Mario KB and check
    whether vector arithmetic captures the same knowledge.

    Rule: Character(?c) ^ coOccursWith(?c, ?g) ^ VideoGame(?g)
          -> appearsIn(?c, ?g)
    Test: vector(coOccursWith) + vector(VideoGame_type) ≈ vector(appearsIn)
    """
    print("\n" + "=" * 60)
    print("Section 8: Rule-Based vs Embedding-Based Reasoning")
    print("=" * 60)

    model = transe_result.model
    model.eval()

    ent_emb = model.entity_representations[0](
        indices=None
    ).detach().cpu().numpy()
    rel_emb = model.relation_representations[0](
        indices=None
    ).detach().cpu().numpy()
    if rel_emb.dtype in (np.complex64, np.complex128):
        rel_emb = np.real(rel_emb)
    if ent_emb.dtype in (np.complex64, np.complex128):
        ent_emb = np.real(ent_emb)

    e2id = training_tf.entity_to_id
    r2id = training_tf.relation_to_id
    id2r = {v: k for k, v in r2id.items()}

    # Define SWRL rule for Mario KB
    swrl_rule = (
        "Character(?c) ^ coOccursWith(?c, ?g) ^ VideoGame(?g) "
        "-> appearsIn(?c, ?g)"
    )
    print(f"\n  SWRL Rule: {swrl_rule}")
    print(f"  Vector test: vector(coOccursWith) + vector(VideoGame) ≈ vector(appearsIn)")

    # Find the relevant relation/entity embeddings
    cooccurs_uri  = "http://esilv.fr/mario#coOccursWith"
    appears_uri   = "http://esilv.fr/mario#appearsIn"
    videogame_uri = "http://esilv.fr/mario#VideoGame"

    results = {"rule": swrl_rule}
    available = True
    for name, uri, is_rel in [
        ("coOccursWith", cooccurs_uri, True),
        ("appearsIn", appears_uri, True),
        ("VideoGame", videogame_uri, False),
    ]:
        mapping = r2id if is_rel else e2id
        if uri not in mapping:
            print(f"    {name} ({uri}) not found in {'relation' if is_rel else 'entity'} index")
            available = False

    if not available:
        # Try with DBpedia predicates as fallback
        print("\n  Trying fallback with DBpedia predicates ...")
        cooccurs_uri = None
        appears_uri  = None
        type_entity  = None

        # Find any two related predicates
        for uri in r2id:
            if "coOccurs" in uri or "sameAs" in uri:
                cooccurs_uri = uri
            if "appearsIn" in uri or "computingPlatform" in uri:
                appears_uri = uri
        for uri in e2id:
            if "VideoGame" in uri or "Software" in uri:
                type_entity = uri

        if cooccurs_uri and appears_uri and type_entity:
            print(f"    Using: {cooccurs_uri.split('/')[-1]}")
            print(f"    Using: {appears_uri.split('/')[-1]}")
            print(f"    Using: {type_entity.split('/')[-1]}")
        else:
            # Use first two relations and first entity as demo
            r_ids = list(r2id.keys())
            e_ids = list(e2id.keys())
            cooccurs_uri = r_ids[0] if r_ids else None
            appears_uri  = r_ids[1] if len(r_ids) > 1 else None
            type_entity  = e_ids[0] if e_ids else None
            print(f"    Using demo relations for illustration")

    if cooccurs_uri and appears_uri:
        r1_id = r2id.get(cooccurs_uri)
        r2_id = r2id.get(appears_uri)

        if r1_id is not None and r2_id is not None:
            r1_vec = rel_emb[r1_id]
            r2_vec = rel_emb[r2_id]

            # Direct relation similarity
            cos_sim = float(np.dot(r1_vec, r2_vec) / (
                np.linalg.norm(r1_vec) * np.linalg.norm(r2_vec) + 1e-8
            ))
            print(f"\n  Cosine sim(r1, r2) = {cos_sim:.4f}")

            # If we have the type entity
            type_uri = videogame_uri if videogame_uri in e2id else type_entity
            if type_uri and type_uri in e2id:
                e_vec = ent_emb[e2id[type_uri]]
                combined = r1_vec + e_vec

                # Normalize
                norms = np.linalg.norm(rel_emb, axis=1, keepdims=True)
                norms[norms == 0] = 1
                rel_normed = rel_emb / norms
                c_norm = combined / (np.linalg.norm(combined) + 1e-8)

                # Find nearest relation to combined vector
                sims = rel_normed @ c_norm
                top5 = np.argsort(-sims)[:5]

                print(f"\n  vector(r1) + vector(type_entity) nearest relations:")
                nearest_rels = []
                for rank, ti in enumerate(top5, 1):
                    tl = id2r[ti].rsplit("/", 1)[-1].rsplit("#", 1)[-1]
                    s = float(sims[ti])
                    print(f"    {rank}. {tl} (sim={s:.4f})")
                    nearest_rels.append({"relation": id2r[ti], "sim": s})

                # Check if r2 (appearsIn) is in top results
                r2_sim = float(sims[r2_id])
                r2_rank = int(np.sum(sims > r2_sim)) + 1
                print(f"\n  Target relation rank: {r2_rank} (sim={r2_sim:.4f})")
                results["vector_arithmetic"] = {
                    "nearest": nearest_rels,
                    "target_rank": r2_rank,
                    "target_sim": r2_sim,
                    "direct_cosine": cos_sim,
                }
            else:
                results["vector_arithmetic"] = {"direct_cosine": cos_sim}

    results["interpretation"] = (
        "The vector arithmetic test checks if the embedding space captures "
        "the logical relationship expressed by the SWRL rule. A perfect match "
        "would mean the embedding learned the same inference. In practice, "
        "KGE models capture distributional patterns which may or may not align "
        "with explicit logical rules. TransE's translational property makes "
        "such analogies more interpretable than bilinear models like ComplEx."
    )

    with open(os.path.join(RES_DIR, "rule_vs_embedding.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)

    return results


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────
def main(skip_size_experiment=False):
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    os.makedirs(RES_DIR, exist_ok=True)

    # 1. Train models (Sections 2-4)
    results, training_tf = train_both_models(device=device)

    # 2. Model comparison (Section 5.1)
    best_name = compare_models(results)
    best_result = results[best_name]["result"]

    # 3. KB size experiment (Section 5.2)
    size_results = None
    if not skip_size_experiment:
        size_results = kb_size_experiment(best_name, device=device)

    # 4. Nearest neighbors (Section 6.1)
    nn_results = nearest_neighbors(best_result, training_tf)

    # 5. t-SNE visualization (Section 6.2)
    tsne_path = tsne_visualization(best_result, training_tf)

    # 6. Relation behavior (Section 6.3)
    rel_analysis = relation_behavior(best_result, training_tf)

    # 7. Rule vs embedding (Section 8)
    transe_result = results["TransE"]["result"]
    rule_results = rule_vs_embedding(transe_result, training_tf)

    # Collect all results
    all_results = {
        "config": CONFIG,
        "model_comparison": {n: d["metrics"] for n, d in results.items()},
        "best_model": best_name,
        "size_experiment": size_results,
        "tsne_plot": tsne_path,
    }
    with open(os.path.join(RES_DIR, "all_results.json"), "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    return all_results


if __name__ == "__main__":
    main()
