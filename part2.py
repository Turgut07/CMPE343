# Part 2: User Variability Modeling
"""
Part 2

Goal:
We look at how many recommendation rounds it takes until a user gives a 5 star.
We call it Tu (1 means they give 5 star on first round).

We do 2 models:
- Geometric (same p for everyone)
- Beta-Geometric (each user has their own p, drawn from Beta(alpha,beta))

Then we split users into 2 groups by popularity of songs they rate 5,
and do simple hypothesis tests to see if Tu is different.
"""

import pandas as pd
import numpy as np
import os
import math


datadir = os.path.join(os.path.dirname(__file__), "..", "data")
resultsdir = os.path.join(os.path.dirname(__file__), "..", "results")


def ensure_results_dir():
    # make results folder if it does not exist
    if os.path.exists(resultsdir) == False:
        os.makedirs(resultsdir)


def load_tracks():
    # read tracks.csv so we can use track_popularity for group split
    path = os.path.join(datadir, "tracks.csv")
    df = pd.read_csv(path)
    return df


def load_ratings():
    # read user_ratings.csv, keep only needed columns
    path = os.path.join(datadir, "user_ratings.csv")
    df = pd.read_csv(path)

    use_cols = ["user_id", "round_idx", "song_id", "rating"]
    df = df[use_cols]
    return df


def merge_ratings_tracks(ratings_df, tracks_df):
    # merge ratings with tracks using song_id == track_id
    merged = pd.merge(
        ratings_df,
        tracks_df,
        left_on="song_id",
        right_on="track_id",
        how="inner"
    )
    return merged


def compute_Tu(ratings_df):
    # Tu is the first time (round) user gives rating 5, but 1-based
    # if user never gives 5, we just dont include them
    firstfive = {}

    # go each rating row, its slow but easy and clear
    for i in range(len(ratings_df)):
        u = ratings_df["user_id"].iloc[i]
        r = ratings_df["rating"].iloc[i]
        t = ratings_df["round_idx"].iloc[i]

        # only 5 stars matter for Tu
        if r == 5:
            # round_idx is 0-based but Tu we want 1.. so +1
            tval = int(t) + 1

            # if user not added yet, just set it
            if u not in firstfive:
                firstfive[u] = tval
            else:
                # keep the earliest 5 star time
                if tval < firstfive[u]:
                    firstfive[u] = tval

        # if not 5, do nothing and continue

    # build dataframe from dict
    rows = []
    for u in firstfive:
        rows.append([u, int(firstfive[u])])

    out = pd.DataFrame(rows, columns=["user_id", "Tu"])
    out = out.sort_values("Tu", ascending=True)
    return out


def fit_geometric(Tu_df):
    """
    Geometric:
      P(T=t) = (1-p)^(t-1) * p

    Mean:
      E[T] = 1/p  ->  p_hat = 1/mean(T)
    """
    tlist = list(Tu_df["Tu"].values)

    # if nothing, return None
    if len(tlist) == 0:
        return None, None

    # compute mean by hand (simple)
    s = 0.0
    for i in range(len(tlist)):
        s = s + float(tlist[i])

    mean_t = s / float(len(tlist))

    # mean should be positive
    if mean_t <= 0:
        return None, None

    p_hat = 1.0 / mean_t
    return p_hat, mean_t


def geometric_pmf(t, p):
    # geometric support starts at t=1
    if t < 1:
        return 0.0

    # formula for pmf
    return ((1.0 - p) ** (t - 1)) * p


def log_beta(a, b):
    # log(B(a,b)) using lgamma so it is stable
    return math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)


def beta_geometric_logpmf(t, alpha, beta):
    """
    Beta-Geometric pmf:
      P(T=t | alpha,beta) = B(alpha+1, beta+t-1) / B(alpha,beta)

    we compute log prob because it is more stable
    """
    # invalid t
    if t < 1:
        return -1e18

    top = log_beta(alpha + 1.0, beta + float(t) - 1.0)
    bot = log_beta(alpha, beta)

    return top - bot


def beta_geometric_loglik(tlist, alpha, beta):
    # sum log pmf for each observed t
    ll = 0.0

    for i in range(len(tlist)):
        t = int(tlist[i])
        ll = ll + beta_geometric_logpmf(t, alpha, beta)

    return ll


def fit_beta_geometric_mle(tlist):
    # grid search then a small refine near the best point
    if len(tlist) == 0:
        return None, None, None

    besta = 1.0
    bestb = 1.0
    bestll = None

    # first scan, step is 0.5
    a_vals = np.arange(0.5, 15.0 + 1e-9, 0.5)
    b_vals = np.arange(0.5, 15.0 + 1e-9, 0.5)

    for a in a_vals:
        for b in b_vals:
            # compute likelihood for this (a,b)
            ll = beta_geometric_loglik(tlist, float(a), float(b))

            # first time init, later compare
            if bestll is None:
                bestll = ll
                besta = float(a)
                bestb = float(b)
            else:
                if ll > bestll:
                    bestll = ll
                    besta = float(a)
                    bestb = float(b)

    # refine near best, smaller step
    a2 = np.arange(max(0.1, besta - 1.0), besta + 1.0 + 1e-9, 0.1)
    b2 = np.arange(max(0.1, bestb - 1.0), bestb + 1.0 + 1e-9, 0.1)

    for a in a2:
        for b in b2:
            ll = beta_geometric_loglik(tlist, float(a), float(b))
            if ll > bestll:
                bestll = ll
                besta = float(a)
                bestb = float(b)

    return besta, bestb, bestll


def make_empirical_table(tlist):
    # build empirical distribution of Tu
    counts = {}

    for i in range(len(tlist)):
        t = int(tlist[i])

        # init key if needed
        if t not in counts:
            counts[t] = 0

        counts[t] = counts[t] + 1

    total = float(len(tlist))

    rows = []
    keys = list(counts.keys())
    keys.sort()

    for k in keys:
        # p_emp is count/total
        rows.append([k, counts[k], counts[k] / total])

    out = pd.DataFrame(rows, columns=["t", "count", "p_emp"])
    return out


def make_geometric_table(emp_table, p_hat):
    # for each observed t, compute geometric model prob
    rows = []

    for i in range(len(emp_table)):
        t = int(emp_table["t"].iloc[i])
        p_model = geometric_pmf(t, p_hat)
        rows.append([t, p_model])

    out = pd.DataFrame(rows, columns=["t", "p_geo"])
    return out


def make_beta_geo_table(emp_table, alpha, beta):
    # for each observed t, compute beta-geometric model prob
    rows = []

    for i in range(len(emp_table)):
        t = int(emp_table["t"].iloc[i])

        # log prob to prob
        lp = beta_geometric_logpmf(t, float(alpha), float(beta))
        p = math.exp(lp)

        rows.append([t, p])

    out = pd.DataFrame(rows, columns=["t", "p_beta_geo"])
    return out


def split_users_by_popularity(ratings_tracks_df, Tu_df):
    # split users by average popularity of songs they rated 5
    # groupA: higher than median, groupB: lower than median
    if "track_popularity" not in ratings_tracks_df.columns:
        return None, None

    popsum = {}
    popcnt = {}

    for i in range(len(ratings_tracks_df)):
        u = ratings_tracks_df["user_id"].iloc[i]
        r = ratings_tracks_df["rating"].iloc[i]

        # only use 5 star ratings to define what user "favorites"
        if r != 5:
            continue

        p = ratings_tracks_df["track_popularity"].iloc[i]
        if pd.isna(p):
            continue

        if u not in popsum:
            popsum[u] = 0.0
            popcnt[u] = 0

        popsum[u] = popsum[u] + float(p)
        popcnt[u] = popcnt[u] + 1

    meanpop = {}
    for u in popsum:
        if popcnt[u] > 0:
            meanpop[u] = popsum[u] / float(popcnt[u])

    # take popularity values only for users that appear in Tu table
    vals = []
    for i in range(len(Tu_df)):
        u = Tu_df["user_id"].iloc[i]
        if u in meanpop:
            vals.append(meanpop[u])

    if len(vals) == 0:
        return None, None

    vals_sorted = sorted(vals)
    mid = len(vals_sorted) // 2

    # median calculation
    if len(vals_sorted) % 2 == 1:
        med = vals_sorted[mid]
    else:
        med = (vals_sorted[mid - 1] + vals_sorted[mid]) / 2.0

    groupA = []
    groupB = []

    # assign each user to a group, store their Tu value
    for i in range(len(Tu_df)):
        u = Tu_df["user_id"].iloc[i]
        t = int(Tu_df["Tu"].iloc[i])

        if u not in meanpop:
            continue

        mp = meanpop[u]

        if mp >= med:
            groupA.append(t)
        else:
            groupB.append(t)

    return groupA, groupB


def normal_cdf(x):
    # normal cdf using erf
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def rankdata_average_ties(x):
    # rank data with average ranks for ties
    x = np.asarray(x)
    n = len(x)

    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(n, dtype=float)

    i = 0
    while i < n:
        j = i

        # find tie block in sorted order
        while j + 1 < n and x[order[j + 1]] == x[order[i]]:
            j = j + 1

        # average rank of positions i..j (1-based ranks)
        avg_rank = 0.5 * ((i + 1) + (j + 1))

        # fill the tie block ranks
        for k in range(i, j + 1):
            ranks[order[k]] = avg_rank

        i = j + 1

    return ranks


def mann_whitney_u_test(a_list, b_list):
    # two-sided mann-whitney u test with normal approx
    a = np.asarray(a_list, dtype=float)
    b = np.asarray(b_list, dtype=float)

    n1 = len(a)
    n2 = len(b)

    if n1 == 0 or n2 == 0:
        return None, None

    # combine then rank
    x = np.concatenate([a, b])
    ranks = rankdata_average_ties(x)

    # sum ranks for group A
    r1 = float(np.sum(ranks[:n1]))

    # U stats
    U1 = r1 - n1 * (n1 + 1) / 2.0
    U2 = n1 * n2 - U1

    # use smaller for two-sided normal approx
    U = min(U1, U2)

    N = n1 + n2
    mu = n1 * n2 / 2.0

    # tie correction
    _, counts = np.unique(x, return_counts=True)
    tie_sum = float(np.sum(counts**3 - counts))

    sigma2 = 0.0
    if N > 1:
        sigma2 = (n1 * n2 / 12.0) * ((N + 1.0) - tie_sum / (N * (N - 1.0)))

    if sigma2 <= 1e-12:
        return float(U), 1.0

    sigma = math.sqrt(sigma2)

    # continuity correction
    z = (U - mu + 0.5) / sigma
    p = 2.0 * (1.0 - normal_cdf(abs(z)))

    if p < 0.0:
        p = 0.0
    if p > 1.0:
        p = 1.0

    return float(U), float(p)


def welch_t_stat(a_list, b_list):
    # compute welch t-stat (p-value not here)
    a = np.asarray(a_list, dtype=float)
    b = np.asarray(b_list, dtype=float)

    n1 = len(a)
    n2 = len(b)

    if n1 < 2:
        return None
    if n2 < 2:
        return None

    m1 = float(np.mean(a))
    m2 = float(np.mean(b))

    v1 = float(np.var(a, ddof=1))
    v2 = float(np.var(b, ddof=1))

    denom = math.sqrt(v1 / n1 + v2 / n2)
    if denom <= 1e-12:
        return 0.0

    return (m1 - m2) / denom


def permutation_test_mean_diff(a_list, b_list, B=3000, seed=123):
    # two-sided permutation test for difference in means
    a = np.asarray(a_list, dtype=float)
    b = np.asarray(b_list, dtype=float)

    n1 = len(a)
    n2 = len(b)

    if n1 == 0 or n2 == 0:
        return None

    obs = abs(float(np.mean(a) - np.mean(b)))
    x = np.concatenate([a, b])

    rng = np.random.default_rng(seed)

    count = 0
    for _ in range(B):
        # shuffle values and split
        perm = rng.permutation(x)
        a2 = perm[:n1]
        b2 = perm[n1:]

        diff = abs(float(np.mean(a2) - np.mean(b2)))
        if diff >= obs:
            count = count + 1

    # add-one smoothing
    p = (count + 1.0) / (B + 1.0)
    return float(p)


def save_csv(name, df):
    # helper to save csv to results folder
    ensure_results_dir()
    path = os.path.join(resultsdir, name)
    df.to_csv(path, index=False)


def print_beta_examples(emp_table, tlist):
    # small "explore alpha,beta" thing, just print a few choices
    # this does not change outputs, its just info like assignment wants
    pairs = [
        (1.0, 1.0),
        (2.0, 8.0),
        (8.0, 2.0),
        (5.0, 5.0),
    ]

    print("\nTrying some different alpha,beta just to see behavior")
    for a, b in pairs:
        ll = beta_geometric_loglik(tlist, a, b)
        tbl = make_beta_geo_table(emp_table, a, b)

        # show first 5 rows (t and p)
        head = tbl.head(5)
        print("alpha =", a, "beta =", b, "loglik =", ll)
        print(head)


def run_part2():
    # load input data
    tracks = load_tracks()
    ratings = load_ratings()

    # compute Tu per user (only users who gave 5 at least once)
    Tu_df = compute_Tu(ratings)

    print("\nusers with at least one 5 star =", len(Tu_df))

    if len(Tu_df) == 0:
        print("no 5 star users so part2 cannot run")
        return

    tlist = list(Tu_df["Tu"].values)

    # geometric part
    print("\nGeometric model")
    p_hat, mean_t = fit_geometric(Tu_df)

    print("mean(Tu) =", mean_t)
    print("p_hat =", p_hat)

    emp = make_empirical_table(tlist)
    geo = make_geometric_table(emp, p_hat)
    show = pd.merge(emp, geo, on="t", how="left")

    print("\nEmpirical vs Geometric first 15 rows")
    print(show.head(15))

    # beta-geometric part
    print("\nBeta-Geometric model")
    alpha_hat, beta_hat, best_ll = fit_beta_geometric_mle(tlist)

    print("alpha_hat =", alpha_hat)
    print("beta_hat =", beta_hat)
    print("best loglik =", best_ll)

    if (alpha_hat + beta_hat) > 0:
        ep = alpha_hat / (alpha_hat + beta_hat)
        print("E[p] =", ep)

    beta_tbl = make_beta_geo_table(emp, alpha_hat, beta_hat)
    show2 = pd.merge(emp, beta_tbl, on="t", how="left")

    print("\nEmpirical vs Beta-Geometric first 15 rows")
    print(show2.head(15))

    # simple exploration for other alpha,beta choices
    print_beta_examples(emp, tlist)

    # hypothesis test part
    print("\nHypothesis test: popular vs less popular")
    ratings_tracks = merge_ratings_tracks(ratings, tracks)
    groupA, groupB = split_users_by_popularity(ratings_tracks, Tu_df)

    # H0: mean Tu same, H1: mean Tu different
    if groupA is None:
        print("cannot split groups maybe popularity missing")
    else:
        print("groupA size =", len(groupA))
        print("groupB size =", len(groupB))

        if len(groupA) > 1 and len(groupB) > 1:
            # mann whitney u is ok when data is skewed
            U_stat, p_u = mann_whitney_u_test(groupA, groupB)
            print("mann-whitney U =", U_stat, "p =", p_u)

            # also show welch t-stat and permutation p (extra info)
            t_stat = welch_t_stat(groupA, groupB)
            p_perm = permutation_test_mean_diff(groupA, groupB, B=3000, seed=123)

            print("welch t-stat =", t_stat)
            print("permutation p for mean diff =", p_perm)

            meanA = sum(groupA) / float(len(groupA))
            meanB = sum(groupB) / float(len(groupB))
            print("mean Tu groupA =", meanA)
            print("mean Tu groupB =", meanB)
        else:
            print("not enough users in groups for tests")

    # save csv outputs (same files as before, so part3 can read)
    pri = pd.DataFrame([[alpha_hat, beta_hat]], columns=["alpha", "beta"])
    save_csv("part2_priors.csv", pri)

    save_csv("part2_Tu.csv", Tu_df)
    save_csv("part2_empirical_vs_geometric.csv", show)
    save_csv("part2_empirical_vs_beta_geometric.csv", show2)

    print("\npart2 done, csv saved in results folder")


def main():
    # main entrypoint
    print("Part 2: User Variability Modeling")
    run_part2()


if __name__ == "__main__":
    main()