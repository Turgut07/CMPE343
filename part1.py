# Part 1: Conditional Probability Modeling

"""
Part 1

We try to estimate how likely a song gets 5* using conditional probs.
Mostly we compute P(5 | feature=value) with Laplace smoothing.

Also we do some simple Bayes inversion like P(feature | 5).
And at the end we export some csv files that Part3 will use.
"""

import pandas as pd
import numpy as np
import os

# extra imports for Part3 csv export
import math
from collections import defaultdict


DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")


def main():
    # main entry just runs part1
    print("Part 1: Conditional Probability Modeling")
    print("TODO: Implement conditional probability calculations")
    print("- Load tracks.csv and ratings.csv")
    print("- Compute P(5* | Artist), P(5* | Year), P(5* | Popularity)")
    print("- Apply Laplace smoothing")
    print("- Use Bayes' rule to invert relationships")
    print("\n--- Running Part 1 implementation ---")
    run_part1()


def ensure_results_dir():
    # we want to save csv outputs so make sure folder exists
    if os.path.exists(RESULTS_DIR) == False:
        os.makedirs(RESULTS_DIR)


def load_tracks():
    # read tracks.csv file
    path = os.path.join(DATA_DIR, "tracks.csv")
    tracks = pd.read_csv(path)
    return tracks


def load_global_ratings():
    # read user_ratings.csv (all users ratings)
    path = os.path.join(DATA_DIR, "user_ratings.csv")
    ratings = pd.read_csv(path)

    # keep only columns we need (easier merges, less confusion)
    use_cols = ["user_id", "round_idx", "song_id", "rating"]
    ratings = ratings[use_cols]
    return ratings


def load_session_ratings(session_index):
    # read my_ratings1.csv / my_ratings2.csv / my_ratings3.csv
    fname = "my_ratings" + str(session_index) + ".csv"
    path = os.path.join(DATA_DIR, fname)
    df = pd.read_csv(path)

    # same columns format as global ratings, so later code is same
    use_cols = ["user_id", "round_idx", "song_id", "rating"]
    df = df[use_cols]
    return df


def merge_ratings_with_tracks(ratings_df, tracks_df):
    # merge ratings with track metadata so we can use artist/genre etc
    # song_id in ratings matches track_id in tracks
    merged = pd.merge(
        ratings_df,
        tracks_df,
        left_on="song_id",
        right_on="track_id",
        how="inner"
    )
    return merged


def add_decade_bin(df):
    # add release_decade col like "1990s", "2000s"
    # we use album_release_year for this
    if "album_release_year" not in df.columns:
        return df

    decs = []

    # go row by row to compute decade
    for i in range(len(df)):
        y = df["album_release_year"].iloc[i]

        # if year missing we keep NaN
        if pd.isna(y):
            decs.append(np.nan)
        else:
            # compute decade by floor division
            y_int = int(y)
            d = (y_int // 10) * 10
            decs.append(str(d) + "s")

    # add the new column
    df["release_decade"] = decs
    return df


def add_popularity_bin(df):
    # make 3 bins from track_popularity: low, mid, high
    # just simple thresholds, nothing fancy
    if "track_popularity" not in df.columns:
        return df

    bins = []

    # loop over each row and choose a bin label
    for i in range(len(df)):
        p = df["track_popularity"].iloc[i]

        # if popularity missing, keep NaN
        if pd.isna(p):
            bins.append(np.nan)
        else:
            # low if <=33
            if p <= 33:
                bins.append("low")
            else:
                # mid if <=66, else high
                if p <= 66:
                    bins.append("mid")
                else:
                    bins.append("high")

    # add pop_bin column
    df["pop_bin"] = bins
    return df


def cond_prob_5(df, feature, alpha):
    # compute P(5 | feature=value) with Laplace smoothing
    # treat rating==5 as "success", others "fail"
    total = {}
    five = {}

    # count how many times each feature value appears
    for i in range(len(df)):
        val = df[feature].iloc[i]

        # if feature value missing, skip it
        if pd.isna(val):
            continue

        r = df["rating"].iloc[i]

        # update total count for this value
        if val not in total:
            total[val] = 0
        total[val] = total[val] + 1

        # if rating is 5, update five count too
        if r == 5:
            if val not in five:
                five[val] = 0
            five[val] = five[val] + 1

    rows = []

    # for each value, compute smoothed probability
    for val in total:
        t = float(total[val])

        # if no 5 entries, f stays 0
        f = 0.0
        if val in five:
            f = float(five[val])

        # Laplace smoothing for Bernoulli:
        # (f + alpha) / (t + 2*alpha)
        p5 = (f + alpha) / (t + 2.0 * alpha)

        # keep row for output table
        rows.append([val, t, f, p5])

    # make dataframe result
    out = pd.DataFrame(rows, columns=[feature, "count", "five", "p_5"])

    # sort so best p_5 is on top
    out = out.sort_values("p_5", ascending=False)
    return out


def cond_prob_5_pair(df, feature1, feature2, alpha, min_count):
    # compute P(5 | f1=value1, f2=value2) with Laplace smoothing
    # we also ignore pairs with too small count (min_count) so it is not random
    total = {}
    five = {}

    # count pairs
    for i in range(len(df)):
        v1 = df[feature1].iloc[i]
        v2 = df[feature2].iloc[i]

        # skip if any missing
        if pd.isna(v1):
            continue
        if pd.isna(v2):
            continue

        key = (v1, v2)
        r = df["rating"].iloc[i]

        # total pair count
        if key not in total:
            total[key] = 0
        total[key] = total[key] + 1

        # if rating is 5, count in five dict
        if r == 5:
            if key not in five:
                five[key] = 0
            five[key] = five[key] + 1

    rows = []

    # compute probability for each pair
    for key in total:
        t = float(total[key])

        # skip if not enough examples
        if t < min_count:
            continue

        # number of 5s for this pair
        f = 0.0
        if key in five:
            f = float(five[key])

        # same smoothing idea
        p5 = (f + alpha) / (t + 2.0 * alpha)

        rows.append([key[0], key[1], t, f, p5])

    out = pd.DataFrame(rows, columns=[feature1, feature2, "count", "five", "p_5"])
    out = out.sort_values("p_5", ascending=False)
    return out


def bayes_feature_given_5(df, feature, alpha):
    # Bayes rule:
    # P(F=f | 5) = P(5 | F=f) * P(F=f) / P(5)
    # we compute P(5|F=f) using our smoothed cond_prob_5 table

    n_total = len(df)
    if n_total == 0:
        return None

    # count how many 5 ratings exist
    n_five = 0
    for i in range(len(df)):
        # check rating each row
        if df["rating"].iloc[i] == 5:
            n_five = n_five + 1

    # if no 5 at all, Bayes doesn't make sense
    if n_five == 0:
        return None

    # compute P(5)
    p5 = n_five / n_total

    # compute table for P(5 | F=f)
    cp = cond_prob_5(df, feature, alpha)

    # compute sum of counts for P(F=f)
    sum_count = 0.0
    for i in range(len(cp)):
        # add count column from table
        sum_count = sum_count + float(cp["count"].iloc[i])

    if sum_count == 0:
        return None

    post = {}

    # now compute unnormalized posterior for each feature value
    for i in range(len(cp)):
        fval = cp[feature].iloc[i]

        # P(5|F=f)
        p_5_given_f = float(cp["p_5"].iloc[i])

        # P(F=f) from counts
        p_f = float(cp["count"].iloc[i]) / sum_count

        # Bayes formula
        val = (p_5_given_f * p_f) / p5
        post[str(fval)] = val

    # normalize so it sums to 1
    s = 0.0
    for k in post:
        s = s + post[k]

    # only normalize if sum is > 0
    if s > 0:
        for k in post:
            post[k] = post[k] / s

    out = pd.Series(post)
    out = out.sort_values(ascending=False)
    return out


def save_anything(name, obj):
    # save a DataFrame or Series into results folder
    ensure_results_dir()
    path = os.path.join(RESULTS_DIR, name + ".csv")

    # if it is dataframe, save normally
    if isinstance(obj, pd.DataFrame):
        obj.to_csv(path, index=False)
    else:
        # if it is a series, save with header
        if isinstance(obj, pd.Series):
            obj.to_csv(path, header=True)


def analyze_global(df_all):
    # global analysis with all users combined
    print("\n=== GLOBAL CONDITIONAL PROBS ===")

    alpha = 1.0
    results = {}

    # list of features we want to check
    need = [
        ("primary_artist_name", "Artist"),
        ("release_decade", "Decade"),
        ("explicit", "Explicit"),
        ("pop_bin", "PopularityBin"),
        ("ab_danceability_value", "Danceability"),
        ("ab_genre_dortmund_value", "Genre"),
        ("ab_timbre_value", "Timbre"),
        ("ab_mood_happy_value", "MoodHappy"),
    ]

    # loop features and compute P(5|feature)
    for col, label in need:
        # check if column exists in dataframe
        if col in df_all.columns:
            print("\nP(5* | " + label + ") top 10:")

            # compute conditional table
            table = cond_prob_5(df_all, col, alpha)

            # show first 10 for debug
            print(table.head(10))

            # store output for saving
            results["global_P5_given_" + col] = table

    print("\n=== GLOBAL FEATURE INTERACTIONS ===")

    # here we do some pair tables (2 features together)
    # min_count helps because many pairs appear only 1 time

    # example: Artist + Genre
    if "primary_artist_name" in df_all.columns:
        if "ab_genre_dortmund_value" in df_all.columns:
            min_count = 3
            print("\nP(5* | Artist, Genre) top 10 (count>=" + str(min_count) + "):")

            # compute pair table
            t2 = cond_prob_5_pair(
                df_all,
                "primary_artist_name",
                "ab_genre_dortmund_value",
                alpha,
                min_count
            )

            print(t2.head(10))
            results["global_P5_given_artist_genre"] = t2

    # example: MoodHappy + Timbre
    if "ab_mood_happy_value" in df_all.columns:
        if "ab_timbre_value" in df_all.columns:
            min_count = 3
            print("\nP(5* | MoodHappy, Timbre) top 10 (count>=" + str(min_count) + "):")

            t3 = cond_prob_5_pair(
                df_all,
                "ab_mood_happy_value",
                "ab_timbre_value",
                alpha,
                min_count
            )

            print(t3.head(10))
            results["global_P5_given_moodhappy_timbre"] = t3

    # example: Genre + Timbre
    if "ab_genre_dortmund_value" in df_all.columns:
        if "ab_timbre_value" in df_all.columns:
            min_count = 3
            print("\nP(5* | Genre, Timbre) top 10 (count>=" + str(min_count) + "):")

            t4 = cond_prob_5_pair(
                df_all,
                "ab_genre_dortmund_value",
                "ab_timbre_value",
                alpha,
                min_count
            )

            print(t4.head(10))
            results["global_P5_given_genre_timbre"] = t4

    # example: PopularityBin + Genre
    if "pop_bin" in df_all.columns:
        if "ab_genre_dortmund_value" in df_all.columns:
            min_count = 3
            print("\nP(5* | PopularityBin, Genre) top 10 (count>=" + str(min_count) + "):")

            t5 = cond_prob_5_pair(
                df_all,
                "pop_bin",
                "ab_genre_dortmund_value",
                alpha,
                min_count
            )

            print(t5.head(10))
            results["global_P5_given_popbin_genre"] = t5

    # example: Explicit + Genre
    if "explicit" in df_all.columns:
        if "ab_genre_dortmund_value" in df_all.columns:
            min_count = 3
            print("\nP(5* | Explicit, Genre) top 10 (count>=" + str(min_count) + "):")

            t6 = cond_prob_5_pair(
                df_all,
                "explicit",
                "ab_genre_dortmund_value",
                alpha,
                min_count
            )

            print(t6.head(10))
            results["global_P5_given_explicit_genre"] = t6

    print("\n=== GLOBAL BAYES (P(Feature | 5*)) ===")

    # which features we want to invert with Bayes
    bayes_list = [
        ("primary_artist_name", "Artist"),
        ("ab_genre_dortmund_value", "Genre"),
        ("ab_timbre_value", "Timbre"),
        ("ab_mood_happy_value", "MoodHappy"),
    ]

    for col, label in bayes_list:
        # only do it if column exists
        if col in df_all.columns:
            post = bayes_feature_given_5(df_all, col, alpha)

            # post can be None if no 5 etc
            if post is not None:
                print("\nP(" + label + " | 5*) top 10:")
                print(post.head(10))
                results["global_P_" + col + "_given_5"] = post

    return results


def analyze_personal(df_me, session_name):
    # analysis for one session csv (like my_ratings1)
    print("\n=== PERSONAL SESSION (" + session_name + ") ===")

    alpha = 1.0
    results = {}

    # not too many features, just some important ones
    need = [
        ("primary_artist_name", "Artist"),
        ("release_decade", "Decade"),
        ("explicit", "Explicit"),
        ("pop_bin", "PopularityBin"),
        ("ab_genre_dortmund_value", "Genre"),
        ("ab_timbre_value", "Timbre"),
        ("ab_mood_happy_value", "MoodHappy"),
    ]

    # loop features and compute probabilities in this session
    for col, label in need:
        if col in df_me.columns:
            print("\nMy P(5* | " + label + "):")

            # compute table
            table = cond_prob_5(df_me, col, alpha)

            # print full table because session is small
            print(table)

            results[session_name + "_P5_given_" + col] = table

    return results


def analyze_group(session_tables, feature, feature_label):
    # group average table: average of session probabilities
    # Pgroup(5|F) = mean over sessions
    group_sum = {}
    group_cnt = {}

    # each item in session_tables is a DataFrame from cond_prob_5
    for table in session_tables:
        # go row by row
        for i in range(len(table)):
            v = table[feature].iloc[i]

            # skip missing feature values
            if pd.isna(v):
                continue

            key = str(v)
            p = float(table["p_5"].iloc[i])

            # init if not exist
            if key not in group_sum:
                group_sum[key] = 0.0
                group_cnt[key] = 0

            # accumulate p values
            group_sum[key] = group_sum[key] + p
            group_cnt[key] = group_cnt[key] + 1

    rows = []
    for key in group_sum:
        # average probability
        avg = group_sum[key] / group_cnt[key]
        rows.append([key, avg])

    out = pd.DataFrame(rows, columns=[feature_label, "p_group_5"])
    out = out.sort_values("p_group_5", ascending=False)
    return out


def run_part1():
    # load datasets
    tracks = load_tracks()
    ratings = load_global_ratings()

    # merge ratings with tracks to get metadata columns
    df_all = merge_ratings_with_tracks(ratings, tracks)

    # add extra derived columns
    df_all = add_popularity_bin(df_all)
    df_all = add_decade_bin(df_all)

    # build 3 sessions dataframes (my_ratings1..3)
    session_dfs = []
    for i in range(1, 4):
        # read session file
        sess = load_session_ratings(i)

        # merge with tracks for metadata
        merged = merge_ratings_with_tracks(sess, tracks)

        # add bins like global
        merged = add_popularity_bin(merged)
        merged = add_decade_bin(merged)

        # store for later
        session_dfs.append(merged)

    # run global analysis tables
    global_results = analyze_global(df_all)

    # run personal analysis on first session
    personal_results = analyze_personal(session_dfs[0], "my_ratings1")

    # group-level average analysis
    print("\n=== GROUP LEVEL (AVG OF 3 SESSIONS) ===")
    group_results = {}

    # group for genre
    if "ab_genre_dortmund_value" in session_dfs[0].columns:
        tables = []

        # compute table for each session
        for s in session_dfs:
            tables.append(cond_prob_5(s, "ab_genre_dortmund_value", 1.0))

        # average them
        g = analyze_group(tables, "ab_genre_dortmund_value", "Genre")

        print("\nPgroup(5* | Genre) top 10:")
        print(g.head(10))

        group_results["group_P5_given_genre"] = g

    # group for timbre
    if "ab_timbre_value" in session_dfs[0].columns:
        tables = []

        # compute table for each session
        for s in session_dfs:
            tables.append(cond_prob_5(s, "ab_timbre_value", 1.0))

        # average them
        g = analyze_group(tables, "ab_timbre_value", "Timbre")

        print("\nPgroup(5* | Timbre) top 10:")
        print(g.head(10))

        group_results["group_P5_given_timbre"] = g

    if "primary_artist_name" in session_dfs[0].columns:
        tables = []
        for s in session_dfs:
            tables.append(cond_prob_5(s, "primary_artist_name", 1.0))
        
        g = analyze_group(tables, "primary_artist_name", "Artist")
        print("\nPgroup(5* | Artist) top 10:")
        print(g.head(10))
        group_results["group_P5_given_artist"] = g

    # 2. GROUP - POPULARITY (BIN)
    if "pop_bin" in session_dfs[0].columns:
        tables = []
        for s in session_dfs:
            tables.append(cond_prob_5(s, "pop_bin", 1.0))
        
        g = analyze_group(tables, "pop_bin", "PopularityBin")
        print("\nPgroup(5* | PopularityBin) top 10:")
        print(g.head(10))
        group_results["group_P5_given_popbin"] = g

    # save all outputs to results/
    for k in global_results:
        save_anything(k, global_results[k])

    for k in personal_results:
        save_anything(k, personal_results[k])

    for k in group_results:
        save_anything(k, group_results[k])

    # export some extra csv that Part3 will use
    export_part3_csvs(df_all, tracks, ratings)

    print("\n[INFO] Part 1 finished. Results saved under results/ folder.")


def export_part3_csvs(df_all, tracks_df, ratings_df):
    """
    Write some csv files so Part3 can run fast.
    It will save:
    - results/part1_feature_priors.csv (feature,value,p_5)
    - results/part1_track_meta.csv (track meta + base score)
    - results/part3_item_sim.csv (simple item-item sim)
    """
    ensure_results_dir()

    # features we want priors for
    feats = [
        "primary_artist_name",
        "ab_genre_dortmund_value",
        "ab_timbre_value",
        "ab_mood_happy_value",
        "release_decade",
        "pop_bin",
        "explicit",
    ]

    alpha = 1.0
    pri_rows = []
    pri_map = {}

    # compute priors for each feature and store as rows
    for feat in feats:
        # skip if feature does not exist
        if feat not in df_all.columns:
            continue

        # compute P(5|feat=value)
        tab = cond_prob_5(df_all, feat, alpha)

        # init feature dict
        pri_map[feat] = {}

        # loop table rows
        for i in range(len(tab)):
            v = tab[feat].iloc[i]

            # skip missing value rows
            if pd.isna(v):
                continue

            p5 = float(tab["p_5"].iloc[i])
            vv = str(v)

            # store as csv row
            pri_rows.append([feat, vv, p5])

            # store in map for quick lookup later
            pri_map[feat][vv] = p5

    # write priors csv for part3
    pri_df = pd.DataFrame(pri_rows, columns=["feature", "value", "p_5"])
    pri_df.to_csv(os.path.join(RESULTS_DIR, "part1_feature_priors.csv"), index=False)

    # now we build track meta csv with a base score
    t = tracks_df.copy()

    # add bins on track data too
    t = add_popularity_bin(t)
    t = add_decade_bin(t)

    rating_count = {}
    five_count = {}

    # count how many ratings each track has (and how many 5)
    for i in range(len(ratings_df)):
        sid = ratings_df["song_id"].iloc[i]
        r = ratings_df["rating"].iloc[i]

        # if song id missing skip
        if pd.isna(sid):
            continue

        sid = str(sid)

        # init counters
        if sid not in rating_count:
            rating_count[sid] = 0
            five_count[sid] = 0

        # increment total rating count
        rating_count[sid] = rating_count[sid] + 1

        # increment five count if rating==5
        if r == 5:
            five_count[sid] = five_count[sid] + 1

    # weights for base score, same logic as in Part3
    w = {
        "primary_artist_name": 0.20,
        "ab_genre_dortmund_value": 0.25,
        "ab_timbre_value": 0.15,
        "ab_mood_happy_value": 0.10,
        "release_decade": 0.15,
        "pop_bin": 0.10,
        "explicit": 0.05,
    }

    meta_rows = []

    # build each track row
    for i in range(len(t)):
        track_id = t["track_id"].iloc[i]

        # skip missing track id
        if pd.isna(track_id):
            continue
        track_id = str(track_id)

        # find track name column
        if "track_name" in t.columns:
            track_name = t["track_name"].iloc[i]
        else:
            if "name" in t.columns:
                track_name = t["name"].iloc[i]
            else:
                track_name = ""

        # fix name if missing
        if pd.isna(track_name):
            track_name = ""
        track_name = str(track_name)

        # get meta fields (if missing use Unknown)
        if "primary_artist_name" in t.columns:
            artist = t["primary_artist_name"].iloc[i]
        else:
            artist = "Unknown"

        if "ab_genre_dortmund_value" in t.columns:
            genre = t["ab_genre_dortmund_value"].iloc[i]
        else:
            genre = "Unknown"

        if "ab_timbre_value" in t.columns:
            timbre = t["ab_timbre_value"].iloc[i]
        else:
            timbre = "Unknown"

        if "ab_mood_happy_value" in t.columns:
            mood = t["ab_mood_happy_value"].iloc[i]
        else:
            mood = "Unknown"

        if "release_decade" in t.columns:
            decade = t["release_decade"].iloc[i]
        else:
            decade = "Unknown"

        if "pop_bin" in t.columns:
            pop_bin = t["pop_bin"].iloc[i]
        else:
            pop_bin = "Unknown"

        if "explicit" in t.columns:
            explicit = t["explicit"].iloc[i]
        else:
            explicit = "Unknown"

        # if any is NaN, replace by Unknown
        if pd.isna(artist):
            artist = "Unknown"
        if pd.isna(genre):
            genre = "Unknown"
        if pd.isna(timbre):
            timbre = "Unknown"
        if pd.isna(mood):
            mood = "Unknown"
        if pd.isna(decade):
            decade = "Unknown"
        if pd.isna(pop_bin):
            pop_bin = "Unknown"
        if pd.isna(explicit):
            explicit = "Unknown"

        # cast them to string so dict lookup works
        artist = str(artist)
        genre = str(genre)
        timbre = str(timbre)
        mood = str(mood)
        decade = str(decade)
        pop_bin = str(pop_bin)
        explicit = str(explicit)

        # get rating count and five count
        rc = rating_count.get(track_id, 0)
        fc = five_count.get(track_id, 0)

        # per-track P(5) with Laplace
        if rc > 0:
            p5_track = (float(fc) + 1.0) / (float(rc) + 2.0)
        else:
            p5_track = 0.20

        # mix feature priors into one number
        prior_mix = 0.0
        prior_mix = prior_mix + w["primary_artist_name"] * float(pri_map.get("primary_artist_name", {}).get(artist, 0.20))
        prior_mix = prior_mix + w["ab_genre_dortmund_value"] * float(pri_map.get("ab_genre_dortmund_value", {}).get(genre, 0.20))
        prior_mix = prior_mix + w["ab_timbre_value"] * float(pri_map.get("ab_timbre_value", {}).get(timbre, 0.20))
        prior_mix = prior_mix + w["ab_mood_happy_value"] * float(pri_map.get("ab_mood_happy_value", {}).get(mood, 0.20))
        prior_mix = prior_mix + w["release_decade"] * float(pri_map.get("release_decade", {}).get(decade, 0.20))
        prior_mix = prior_mix + w["pop_bin"] * float(pri_map.get("pop_bin", {}).get(pop_bin, 0.20))
        prior_mix = prior_mix + w["explicit"] * float(pri_map.get("explicit", {}).get(explicit, 0.20))

        # read popularity numeric value
        pop = 0.0
        if "track_popularity" in t.columns:
            val = t["track_popularity"].iloc[i]
            if pd.isna(val) == False:
                try:
                    pop = float(val)
                except:
                    pop = 0.0

        # clamp popularity to 0..100
        if pop < 0:
            pop = 0.0
        if pop > 100:
            pop = 100.0

        # normalize popularity to 0..1
        pop_norm = pop / 100.0

        # base score formula (same as part3 uses)
        base = 0.62 * p5_track + 0.28 * prior_mix + 0.10 * pop_norm

        # store row for csv
        meta_rows.append([
            track_id, track_name,
            artist, genre, timbre, mood, decade, pop_bin, explicit,
            pop, p5_track, base, rc
        ])

    # build dataframe and write it
    meta_df = pd.DataFrame(
        meta_rows,
        columns=[
            "track_id", "track_name",
            "primary_artist_name", "ab_genre_dortmund_value", "ab_timbre_value", "ab_mood_happy_value",
            "release_decade", "pop_bin", "explicit",
            "track_popularity", "p5_track", "base", "rating_count"
        ]
    )
    meta_df.to_csv(os.path.join(RESULTS_DIR, "part1_track_meta.csv"), index=False)

    # export item-item similarity for part3
    export_item_sim_csv(ratings_df)


def export_item_sim_csv(ratings_df):
    """
    Simple co-like item similarity:
    - for each user take songs with rating>=4
    - count co-occurrence
    - sim = co / sqrt(freqA * freqB)
    - keep top 40 neighbors per item
    """
    # sort so user sequences are in order (not super needed but ok)
    df = ratings_df.sort_values(["user_id", "round_idx"], ascending=True)

    userlikes = defaultdict(list)
    freq = defaultdict(int)

    # build user -> liked items list
    for i in range(len(df)):
        u = df["user_id"].iloc[i]
        sid = df["song_id"].iloc[i]
        r = df["rating"].iloc[i]

        # skip missing
        if pd.isna(u) or pd.isna(sid):
            continue

        # consider rating>=4 as like
        if r >= 4:
            u = str(u)
            sid = str(sid)

            # add item to that user list
            userlikes[u].append(sid)

            # track frequency of likes for that item
            freq[sid] = freq[sid] + 1

    co = defaultdict(lambda: defaultdict(int))

    RECENT_CAP = 50

    # compute co-occurrence counts for each user
    for u in userlikes:
        items = userlikes[u]

        # cap last 50 likes so it does not explode
        if len(items) > RECENT_CAP:
            items = items[-RECENT_CAP:]

        # pairwise counts
        for i in range(len(items)):
            a = items[i]
            for j in range(i + 1, len(items)):
                b = items[j]

                # ignore same item
                if a == b:
                    continue

                # increment both directions
                co[a][b] = co[a][b] + 1
                co[b][a] = co[b][a] + 1

    out_rows = []
    TOP_NEIGH = 40

    # for each item, compute similarity with neighbors
    for a in co:
        lst = []

        # compute sim for each neighbor
        for b in co[a]:
            c = float(co[a][b])
            fa = float(freq.get(a, 1))
            fb = float(freq.get(b, 1))

            # cosine-like normalization
            sim = c / max(math.sqrt(fa * fb), 1e-9)
            lst.append((b, sim))

        # sort neighbors by similarity desc
        lst.sort(key=lambda x: x[1], reverse=True)

        # keep top 40 neighbors
        for b, sim in lst[:TOP_NEIGH]:
            out_rows.append([a, b, sim])

    # write csv output
    sim_df = pd.DataFrame(out_rows, columns=["track_id", "neighbor_id", "sim"])
    sim_df.to_csv(os.path.join(RESULTS_DIR, "part3_item_sim.csv"), index=False)


if __name__ == "__main__":
    main()