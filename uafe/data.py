"""Synthetic reference representations d_i (Phase I, step 1).

generate_distribution and generate_mixed_sparsity_distribution are copied unchanged
from the draft notebook. On their own they give every feature the same expected
prevalence, so a prevalence threshold has nothing to separate. generate_population
therefore stacks several generate_distribution blocks, each with its own sparsity and
membership, to produce a long-tailed, group-structured population:

  shared tiers      held across the whole population, from very common to uncommon
  group features    common *within* one subgroup, absent elsewhere -> globally rare
                    when the subgroup is small
  idiosyncratic     each held by one or two people

arrival_feature_names adds features that appear only among new arrivals (Phase III).
"""
from dataclasses import dataclass

import numpy as np


# --------------------------------------------------------------------------------------
# From the draft notebook (unchanged)
# --------------------------------------------------------------------------------------

def generate_mixed_sparsity_distribution(length=10, num_ref_reps=10):
    distributions_with_pooled_populations = []
    # 1. Determine the length of each binary representation (number of bits needed)
    # To represent the number 'length' in binary, we check how many bits are required.
    max_range = 2**length

    for j in range(0,3):
      # 2. Build the full pool of unique binary vectors up to 'length'
      full_ref_reps = []
      for i in range(max_range):
          # Convert number to simple binary string, pad with zeros, and convert to integer list
          bin_str = bin(i)[2:].zfill(length)
          vector = [int(bit) for bit in bin_str]
          full_ref_reps.append(vector)

      # Print the size of the generated unique vector pool
      print(f"Length of full_ref_reps: {len(full_ref_reps)}")

      # 3. Randomly choose indices from our pool to build the ref_reps
      # (Using a default of 10 vectors if not specified)
      chosen_indices = np.random.randint(0, len(full_ref_reps), size=num_ref_reps)
      ref_reps = np.array([full_ref_reps[idx] for idx in chosen_indices])

      # 4. Compute pooled population
      pooled_population = ref_reps.sum(axis=0)

      distributions_with_pooled_populations.append((ref_reps, pooled_population))

    return distributions_with_pooled_populations


#num_vectors := number of reference representations
#length := total number of distinct reported features
#number of 1s in each vector := specific features reported by individual i in reference representation i
def generate_distribution(num_vectors=100, length=1000, sparsity=0.02):
    #create reference representations
    vectors = []
    for _ in range(num_vectors):
        while True:
            vec = np.random.choice([0, 1], size=length, p=[1 - sparsity, sparsity])
            sum_val = np.sum(vec)
            if 0 < sum_val < length:
                vectors.append(vec)
                break
    vectors = np.array(vectors)

    # guarantee that each position (column) has at least one '1'
    pooled = vectors.sum(axis=0)
    zero_cols = np.where(pooled == 0)[0]

    # for any column containing all zeros, pick a random vector to set that index to 1
    for col in zero_cols:
        random_vec_idx = np.random.randint(num_vectors)
        vectors[random_vec_idx, col] = 1

    return vectors


# --------------------------------------------------------------------------------------
# Heterogeneous population
# --------------------------------------------------------------------------------------

# (name, number of features, sparsity) for features held across the whole population
DEFAULT_SHARED_TIERS = (
    ("core", 10, 0.6),
    ("frequent", 20, 0.3),
    ("common", 40, 0.12),
    ("uncommon", 80, 0.04),
)
# subgroup -> share of the population
DEFAULT_GROUPS = {"A": 0.60, "B": 0.25, "C": 0.10, "D": 0.05}


@dataclass
class Population:
    X: np.ndarray              # (N, L) binary reference representations d_i
    y: np.ndarray              # (N,) synthetic outcome for the downstream task
    group: np.ndarray          # (N,) subgroup label
    feature_names: np.ndarray  # (L,)
    feature_kind: np.ndarray   # (L,) "shared" | "group" | "idiosyncratic"

    def __len__(self):
        return len(self.X)

    def subset(self, idx):
        return Population(self.X[idx], self.y[idx], self.group[idx],
                          self.feature_names, self.feature_kind)


def generate_population(n=500, groups=DEFAULT_GROUPS, shared_tiers=DEFAULT_SHARED_TIERS,
                        n_group_features=30, group_sparsity=0.25,
                        n_idiosyncratic=300, idiosyncratic_sparsity=0.004,
                        label_noise=0.5, seed=None):
    """Draw N individuals with heterogeneous binary feature vectors and an outcome y.

    The notebook generators use the global np.random state, so `seed` seeds that.
    """
    if seed is not None:
        np.random.seed(seed)

    names = list(groups)
    sizes = np.floor(np.array([groups[g] for g in names]) * n).astype(int)
    sizes[0] += n - sizes.sum()
    if (sizes < 1).any():
        raise ValueError("every group needs at least one member; increase n")
    group = np.repeat(names, sizes)

    blocks, feature_names, feature_kind = [], [], []

    for tier, length, sparsity in shared_tiers:
        blocks.append(generate_distribution(num_vectors=n, length=length, sparsity=sparsity))
        feature_names += [f"{tier}_{k:02d}" for k in range(length)]
        feature_kind += ["shared"] * length

    for g, size in zip(names, sizes):
        block = np.zeros((n, n_group_features), dtype=int)
        block[group == g] = generate_distribution(num_vectors=size, length=n_group_features,
                                                  sparsity=group_sparsity)
        blocks.append(block)
        feature_names += [f"group{g}_{k:02d}" for k in range(n_group_features)]
        feature_kind += ["group"] * n_group_features

    blocks.append(generate_distribution(num_vectors=n, length=n_idiosyncratic,
                                        sparsity=idiosyncratic_sparsity))
    feature_names += [f"idio_{k:03d}" for k in range(n_idiosyncratic)]
    feature_kind += ["idiosyncratic"] * n_idiosyncratic

    X = np.hstack(blocks).astype(int)
    feature_kind = np.array(feature_kind)

    perm = np.random.permutation(n)
    X, group = X[perm], group[perm]
    y = generate_labels(X, feature_kind, noise=label_noise)
    return Population(X, y, group, np.array(feature_names), feature_kind)


def generate_labels(X, feature_kind, noise=0.5, active_share=0.5):
    """Binary outcome driven by a sparse linear score.

    Half of the shared and group-specific features carry signal (group features with a
    larger weight), idiosyncratic features carry none, plus Gaussian noise scaled by
    `noise` relative to the score's spread. Thresholded at the median, so classes are
    balanced. Dropping a subgroup's features hurts accuracy for that subgroup.
    """
    L = X.shape[1]
    w = np.zeros(L)
    for kind, scale in (("shared", 1.0), ("group", 2.0)):
        idx = np.where(feature_kind == kind)[0]
        active = np.random.rand(len(idx)) < active_share
        w[idx] = np.random.normal(0, scale, len(idx)) * active
    score = X @ w
    score = score + np.random.normal(0, noise * score.std(), len(X))
    return (score > np.median(score)).astype(int)


def arrival_feature_names(arrivals, n_unseen=100, unseen_sparsity=0.015, seed=None):
    """Feature names reported by each new arrival, including features outside the reference.

    In practice new individuals usually report some features no one in the pooled
    population reported. Each arrival keeps its generated features and reports each of
    `n_unseen` brand-new names with probability `unseen_sparsity`. With the defaults that
    is 1.5 per arrival on average, and about 78% of arrivals report at least one.
    generate_distribution is not used here because it forces at least one per arrival.
    """
    if seed is not None:
        np.random.seed(seed)
    unseen = (np.random.rand(len(arrivals), n_unseen) < unseen_sparsity).astype(int)
    unseen_names = np.array([f"unseen_{k:03d}" for k in range(n_unseen)])
    return [arrivals.feature_names[x == 1].tolist() + unseen_names[u == 1].tolist()
            for x, u in zip(arrivals.X, unseen)]
