"""
Part 3

This file reads the csv outputs from Part1 + Part2 and uses them for recommender logic.

Needed files (in results/):
- part1_feature_priors.csv : feature,value,p_5
- part1_track_meta.csv     : track_id, track_name, ..., base, rating_count
- part2_priors.csv         : alpha,beta
- part3_item_sim.csv       : track_id, neighbor_id, sim (optional but if exists we use)

We have 2 models:
Model A: more personalized, mostly deterministic ranking
Model B: more global-first, but has 1 small sampling slot sometimes (a bit explore)
"""

import os
import csv
import math
import random
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Any


BASE_DIR = os.path.dirname(__file__)
RESULTS_DIR = os.path.join(BASE_DIR, "..", "results")

P1_PRI_PATH = os.path.join(RESULTS_DIR, "part1_feature_priors.csv")
P1_META_PATH = os.path.join(RESULTS_DIR, "part1_track_meta.csv")
P2_PRI_PATH = os.path.join(RESULTS_DIR, "part2_priors.csv")
SIM_PATH = os.path.join(RESULTS_DIR, "part3_item_sim.csv")


LIKE_TH = 4
DISLIKE_TH = 2

MAX_SAME_ARTIST = 2

TOP_FEAT = 220
GLOBAL_BACKFILL = 2000
POOL_B = 900

WB_A = 3.0
WCF_A = 4.0
WPRI_A = 2.0


_loaded = False

track_metadata: Dict[str, Dict[str, Any]] = {}
global_ranking: List[str] = []

tracks_by_artist = defaultdict(list)
tracks_by_genre = defaultdict(list)
tracks_by_timbre = defaultdict(list)
tracks_by_decade = defaultdict(list)
tracks_by_popbin = defaultdict(list)

priors_by_feature = defaultdict(dict)
item_similarity = defaultdict(list)

alpha_prior = 4.33
beta_prior = 6.93


# small helper for reading csv as dict list
def _read_csv_dicts(path: str) -> List[Dict[str, str]]:
    rows = []
    # open file and read each row as a dict
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            # add row dict to list
            rows.append(r)
    return rows


# small helper for safe string (student way)
def _safe_str(x: Any) -> str:
    if x is None:
        return ""
    return str(x).strip()


# platform can send different id keys, so we try a few
def parse_track_id(row: Dict[str, Any]) -> str:
    # check common keys one by one
    for key in ["spotify_id", "song_id", "track_id", "id"]:
        if key in row:
            if row[key] is not None:
                s = _safe_str(row[key])
                # if not empty we accept it
                if s != "":
                    return s
    return ""


# load all global stuff once so query is fast
def init_globals():
    global _loaded
    global alpha_prior
    global beta_prior

    # if already loaded, do nothing
    if _loaded == True:
        return

    # read track meta csv
    if os.path.exists(P1_META_PATH) == True:
        rows = _read_csv_dicts(P1_META_PATH)

        # loop every track row
        for r in rows:
            tid = _safe_str(r.get("track_id", ""))

            # if no id, skip this row
            if tid == "":
                continue

            # read fields, if empty -> Unknown
            name = _safe_str(r.get("track_name", ""))

            artist = _safe_str(r.get("primary_artist_name", "Unknown"))
            if artist == "":
                artist = "Unknown"

            genre = _safe_str(r.get("ab_genre_dortmund_value", "Unknown"))
            if genre == "":
                genre = "Unknown"

            timbre = _safe_str(r.get("ab_timbre_value", "Unknown"))
            if timbre == "":
                timbre = "Unknown"

            mood = _safe_str(r.get("ab_mood_happy_value", "Unknown"))
            if mood == "":
                mood = "Unknown"

            decade = _safe_str(r.get("release_decade", "Unknown"))
            if decade == "":
                decade = "Unknown"

            popbin = _safe_str(r.get("pop_bin", "Unknown"))
            if popbin == "":
                popbin = "Unknown"

            explicit = _safe_str(r.get("explicit", "Unknown"))
            if explicit == "":
                explicit = "Unknown"

            # read numeric fields (dataset should be clean, but i keep it simple)
            base_score = 0.0
            try:
                base_score = float(r.get("base", 0.0))
            except:
                base_score = 0.0

            p5_track = 0.20
            try:
                p5_track = float(r.get("p5_track", 0.20))
            except:
                p5_track = 0.20

            rating_count = 0
            try:
                rating_count = int(float(r.get("rating_count", 0)))
            except:
                rating_count = 0

            # store meta into dict
            track_metadata[tid] = {
                "name": name,
                "artist": artist,
                "genre": genre,
                "timbre": timbre,
                "mood": mood,
                "decade": decade,
                "popbin": popbin,
                "explicit": explicit,
                "base": base_score,
                "p5_track": p5_track,
                "rating_count": rating_count
            }

        # make global ranking by base score
        # this is like "most safe/popular" list
        global_sorted = sorted(track_metadata.keys(), key=lambda t: track_metadata[t]["base"], reverse=True)
        for tid in global_sorted:
            global_ranking.append(tid)

        # build indexes for fast recall by feature
        for tid in global_ranking:
            m = track_metadata[tid]

            # add tid under its feature value
            tracks_by_artist[m["artist"]].append(tid)
            tracks_by_genre[m["genre"]].append(tid)
            tracks_by_timbre[m["timbre"]].append(tid)
            tracks_by_decade[m["decade"]].append(tid)
            tracks_by_popbin[m["popbin"]].append(tid)

    # read feature priors (from part1)
    if os.path.exists(P1_PRI_PATH) == True:
        rows = _read_csv_dicts(P1_PRI_PATH)

        # go row by row and fill priors map
        for r in rows:
            feat = _safe_str(r.get("feature", ""))
            val = _safe_str(r.get("value", ""))

            # if missing, skip it
            if feat == "" or val == "":
                continue

            p5 = 0.20
            try:
                p5 = float(r.get("p_5", 0.20))
            except:
                p5 = 0.20

            # store p(5) for this feature/value
            priors_by_feature[feat][val] = p5

    # read alpha beta (from part2)
    if os.path.exists(P2_PRI_PATH) == True:
        rows = _read_csv_dicts(P2_PRI_PATH)

        # just take first row
        if len(rows) > 0:
            try:
                alpha_prior = float(rows[0].get("alpha", alpha_prior))
            except:
                # ignore if weird
                pass

            try:
                beta_prior = float(rows[0].get("beta", beta_prior))
            except:
                pass

    # read item similarity file (optional but nice)
    if os.path.exists(SIM_PATH) == True:
        rows = _read_csv_dicts(SIM_PATH)
        tmp = defaultdict(list)

        # fill tmp first
        for r in rows:
            a = _safe_str(r.get("track_id", ""))
            b = _safe_str(r.get("neighbor_id", ""))

            # skip if no ids
            if a == "" or b == "":
                continue

            sim = 0.0
            try:
                sim = float(r.get("sim", 0.0))
            except:
                sim = 0.0

            tmp[a].append((b, sim))

        # now sort each neighbor list
        for a in tmp:
            tmp[a].sort(key=lambda x: x[1], reverse=True)
            item_similarity[a] = tmp[a]

    # mark loaded
    _loaded = True


# build some user info from the session ratings list
def build_user_profile(song_ratings):
    seen_tracks = set()
    liked_tracks = []
    disliked_artists = set()

    # store counts for top features
    pref_counts = {
        "artist": Counter(),
        "genre": Counter(),
        "timbre": Counter(),
        "mood": Counter(),
        "decade": Counter(),
        "popbin": Counter(),
    }

    # go through each rating row from platform
    for row in song_ratings:
        tid = parse_track_id(row)

        # skip if no id
        if tid == "":
            continue

        # remember seen, so we don't recommend it again
        seen_tracks.add(tid)

        # if we don't know metadata, skip preference learning
        if tid not in track_metadata:
            continue

        rating = 0
        try:
            rating = int(float(row.get("rating", 0)))
        except:
            rating = 0

        meta = track_metadata[tid]

        # if user likes it, count the features
        if rating >= LIKE_TH:
            liked_tracks.append(tid)

            pref_counts["artist"][meta["artist"]] += 1
            pref_counts["genre"][meta["genre"]] += 1
            pref_counts["timbre"][meta["timbre"]] += 1
            pref_counts["mood"][meta["mood"]] += 1
            pref_counts["decade"][meta["decade"]] += 1
            pref_counts["popbin"][meta["popbin"]] += 1

        # if user dislikes, we mark the artist (avoid it later)
        if rating <= DISLIKE_TH:
            disliked_artists.add(meta["artist"])

    return liked_tracks, seen_tracks, disliked_artists, pref_counts


# compute session-only P(5|feature) from the current session
def session_prior_probs(song_ratings, meta_key):
    total = Counter()
    five = Counter()

    # loop each rated song in session
    for row in song_ratings:
        tid = parse_track_id(row)

        # if unknown track, skip
        if tid not in track_metadata:
            continue

        rating = 0
        try:
            rating = int(float(row.get("rating", 0)))
        except:
            # if weird rating, just skip
            continue

        value = track_metadata[tid].get(meta_key, "Unknown")
        if value is None or value == "":
            value = "Unknown"

        # count total for this feature value
        total[value] += 1

        # count five-star for this feature value
        if rating == 5:
            five[value] += 1

    # Laplace smoothing for session
    a = 1.0
    out = {}

    # make probability map
    for v in total:
        out[v] = (float(five[v]) + a) / (float(total[v]) + 2.0 * a)

    return out


# blend global priors + session priors (session small -> global more)
def blend_global_and_session(global_map, session_map, session_counts, k=10.0):
    out = dict(global_map)

    # loop each value we saw in session
    for v in session_map:
        n = float(session_counts.get(v, 0))

        # lambda grows with n
        lam = n / (n + k)

        # fallback global probability
        pg = float(global_map.get(v, 0.20))

        # blend them
        out[v] = lam * float(session_map[v]) + (1.0 - lam) * pg

    return out


# estimate user-level probability of giving 5 based on Beta prior (part2)
def estimate_user_probability(song_ratings):
    five = 0
    n = 0

    # go over session ratings and count 5s
    for row in song_ratings:
        rating = 0
        try:
            rating = int(float(row.get("rating", 0)))
        except:
            continue

        n += 1
        if rating == 5:
            five += 1

    # posterior mean
    return (alpha_prior + five) / (alpha_prior + beta_prior + n)


# simple softmax helper (we keep it stable with max trick)
def softmax(scores, temp):
    if len(scores) == 0:
        return []

    mx = max(scores)
    exps = []

    # compute exp for each score
    for s in scores:
        exps.append(math.exp((s - mx) / max(temp, 1e-9)))

    total = sum(exps)

    # fallback if something is bad
    if total <= 0:
        probs = []
        for _ in scores:
            probs.append(1.0 / len(scores))
        return probs

    probs = []
    for e in exps:
        probs.append(e / total)
    return probs


# weighted sampling without replacement, simple manual way
def weighted_sample_without_replacement(items, probs, k):
    chosen = []
    pool_items = list(items)
    pool_probs = list(probs)

    # pick k times
    for _ in range(k):
        if len(pool_items) == 0:
            break

        s = sum(pool_probs)

        # if probs sum is weird, just random pick
        if s <= 0:
            idx = random.randint(0, len(pool_items) - 1)
        else:
            # roulette wheel sampling
            r = random.random() * s
            acc = 0.0
            idx = 0

            for i in range(len(pool_probs)):
                acc += pool_probs[i]
                if acc >= r:
                    idx = i
                    break

        # add selected item
        chosen.append(pool_items[idx])

        # remove it from pool
        pool_items.pop(idx)
        pool_probs.pop(idx)

    return chosen


# Model A: personalized and mostly deterministic
def query_model_a(song_ratings, topk=5):
    init_globals()

    # if no tracks or topk wrong, nothing to do
    if topk <= 0:
        return []
    if len(track_metadata) == 0:
        return []

    liked_tracks, seen_tracks, disliked_artists, pref_counts = build_user_profile(song_ratings)

    # if user has no likes, fallback to model b
    if len(liked_tracks) == 0:
        return query_model_b(song_ratings, topk)

    # we blend global priors + session priors for some features
    mapping = [
        ("primary_artist_name", "artist"),
        ("ab_genre_dortmund_value", "genre"),
        ("ab_timbre_value", "timbre"),
        ("release_decade", "decade"),
        ("pop_bin", "popbin"),
    ]

    blended = {}

    # build blended priors for each meta key
    for feat_key, meta_key in mapping:
        global_map = priors_by_feature.get(feat_key, {})
        sess_map = session_prior_probs(song_ratings, meta_key)
        blended[meta_key] = blend_global_and_session(global_map, sess_map, pref_counts[meta_key], k=10.0)

    candidate_scores = defaultdict(float)

    # helper to safely add candidates
    def add_candidate(tid, w):
        # skip empty or already seen
        if tid == "":
            return
        if tid in seen_tracks:
            return

        # must exist in our metadata
        if tid not in track_metadata:
            return

        # skip disliked artist
        if track_metadata[tid]["artist"] in disliked_artists:
            return

        # add weight
        candidate_scores[tid] += float(w)

    # 1) CF neighbors from liked tracks
    for i in range(len(liked_tracks)):
        tid = liked_tracks[i]

        # make a small position weight (later likes slightly higher)
        w_like = 1.0 + 0.8 * (i + 1) / max(len(liked_tracks), 1)

        # loop over neighbors
        for nb, sim in item_similarity.get(tid, []):
            add_candidate(nb, WCF_A * float(sim) * w_like)

    # 2) expand by top features (artists/genres/etc)
    top_artists = []
    for pair in pref_counts["artist"].most_common(2):
        top_artists.append(pair[0])

    top_genres = []
    for pair in pref_counts["genre"].most_common(2):
        top_genres.append(pair[0])

    top_timbres = []
    for pair in pref_counts["timbre"].most_common(1):
        top_timbres.append(pair[0])

    top_decades = []
    for pair in pref_counts["decade"].most_common(1):
        top_decades.append(pair[0])

    top_popbins = []
    for pair in pref_counts["popbin"].most_common(1):
        top_popbins.append(pair[0])

    # add from artist index
    for a in top_artists:
        # loop top tracks of that artist
        for tid in tracks_by_artist.get(a, [])[:TOP_FEAT]:
            add_candidate(tid, 1.6)

    # add from genre index
    for g in top_genres:
        for tid in tracks_by_genre.get(g, [])[:TOP_FEAT]:
            add_candidate(tid, 1.1)

    # add from timbre index
    for t in top_timbres:
        for tid in tracks_by_timbre.get(t, [])[:TOP_FEAT]:
            add_candidate(tid, 0.9)

    # add from decade index
    for d in top_decades:
        for tid in tracks_by_decade.get(d, [])[:TOP_FEAT]:
            add_candidate(tid, 0.6)

    # add from popbin index
    for p in top_popbins:
        for tid in tracks_by_popbin.get(p, [])[:TOP_FEAT]:
            add_candidate(tid, 0.5)

    # 3) global backfill (just in case recall is small)
    for tid in global_ranking[:GLOBAL_BACKFILL]:
        add_candidate(tid, 0.25)

    # if still no candidates, fallback
    if len(candidate_scores) == 0:
        return query_model_b(song_ratings, topk)

    scored = []

    # compute final score for each candidate
    for tid in candidate_scores:
        meta = track_metadata[tid]
        base = float(meta["base"])

        # compute user_prob from blended priors
        user_prob = 0.0
        user_prob += 0.35 * float(blended["artist"].get(meta["artist"], 0.20))
        user_prob += 0.35 * float(blended["genre"].get(meta["genre"], 0.20))
        user_prob += 0.15 * float(blended["timbre"].get(meta["timbre"], 0.20))
        user_prob += 0.10 * float(blended["decade"].get(meta["decade"], 0.20))
        user_prob += 0.05 * float(blended["popbin"].get(meta["popbin"], 0.20))

        # final score mix
        score = float(candidate_scores[tid]) + (WB_A * base) + (WPRI_A * user_prob)
        scored.append((tid, score))

    # sort candidates by score desc
    scored.sort(key=lambda x: x[1], reverse=True)

    out = []
    used_artist = Counter()

    # take topk but limit per artist
    for tid, _ in scored:
        a = track_metadata[tid]["artist"]

        # if too many of same artist, skip
        if used_artist[a] >= MAX_SAME_ARTIST:
            continue

        out.append((tid, track_metadata[tid]["name"]))
        used_artist[a] += 1

        # stop if enough
        if len(out) >= topk:
            break

    # fill from global ranking if still not enough
    if len(out) < topk:
        have = set()
        for x in out:
            have.add(x[0])

        # loop global list
        for tid in global_ranking:
            if len(out) >= topk:
                break

            # skip seen or already chosen
            if tid in have:
                continue
            if tid in seen_tracks:
                continue

            # skip disliked artists
            if track_metadata[tid]["artist"] in disliked_artists:
                continue

            a = track_metadata[tid]["artist"]
            if used_artist[a] >= MAX_SAME_ARTIST:
                continue

            out.append((tid, track_metadata[tid]["name"]))
            used_artist[a] += 1
            have.add(tid)

    return out[:topk]


# Model B: global-first + small exploration sometimes
def query_model_b(song_ratings, topk=5):
    init_globals()

    if topk <= 0:
        return []
    if len(track_metadata) == 0:
        return []

    liked_tracks, seen_tracks, disliked_artists, pref_counts = build_user_profile(song_ratings)

    # helper to check if tid is allowed to recommend
    def ok_track(tid):
        if tid == "":
            return False
        if tid in seen_tracks:
            return False
        if tid not in track_metadata:
            return False
        if track_metadata[tid]["artist"] in disliked_artists:
            return False

        # rated universe: we only want tracks that have rating_count > 0
        if int(track_metadata[tid].get("rating_count", 0)) <= 0:
            return False

        return True

    # 1) take a global pool from global_ranking
    global_pool = []

    # go in order and add ok tracks
    for tid in global_ranking:
        if ok_track(tid) == True:
            global_pool.append(tid)

            # stop at pool size
            if len(global_pool) >= POOL_B:
                break

    if len(global_pool) == 0:
        return []

    # 2) recall injection (neighbors + feature recall)
    best_cf = {}

    CF_NEIGH_PER_LIKE = 35
    CF_RECALL_CAP = 350

    # add neighbors from liked tracks
    for lid in liked_tracks:
        neigh = item_similarity.get(lid, [])

        # take top neighbors per liked item
        for nb, sim in neigh[:CF_NEIGH_PER_LIKE]:
            if ok_track(nb) == False:
                continue

            s = float(sim)

            # keep best sim if exists
            if nb not in best_cf:
                best_cf[nb] = s
            else:
                if s > best_cf[nb]:
                    best_cf[nb] = s

            # cap recall dict size
            if len(best_cf) >= CF_RECALL_CAP:
                break

        if len(best_cf) >= CF_RECALL_CAP:
            break

    # get favorite feature values
    top_artists = []
    for pair in pref_counts["artist"].most_common(1):
        top_artists.append(pair[0])

    top_genres = []
    for pair in pref_counts["genre"].most_common(1):
        top_genres.append(pair[0])

    top_timbres = []
    for pair in pref_counts["timbre"].most_common(1):
        top_timbres.append(pair[0])

    top_decades = []
    for pair in pref_counts["decade"].most_common(1):
        top_decades.append(pair[0])

    # helper to add from feature index maps
    def add_from_index(index_map, keys, cap_each):
        # loop keys like artist/genre
        for k in keys:
            # take top items from that index list
            for tid in index_map.get(k, [])[:cap_each]:
                if ok_track(tid) == True:
                    if tid not in best_cf:
                        # add with sim=0, so it is recall but not CF strong
                        best_cf[tid] = 0.0

    add_from_index(tracks_by_artist, top_artists, 120)
    add_from_index(tracks_by_genre, top_genres, 150)
    add_from_index(tracks_by_timbre, top_timbres, 120)
    add_from_index(tracks_by_decade, top_decades, 90)

    # merge global pool + recall items into one pool
    pool = []
    used = set()

    # add global first
    for tid in global_pool:
        if tid not in used:
            pool.append(tid)
            used.add(tid)

    # add recall items next
    for tid in best_cf:
        if tid not in used:
            pool.append(tid)
            used.add(tid)

    # cap pool size for speed
    if len(pool) > 1200:
        pool = pool[:1200]

    # favorites for small bonus scoring
    fav_genres = set()
    for pair in pref_counts["genre"].most_common(2):
        fav_genres.add(pair[0])

    fav_timbre = None
    if len(pref_counts["timbre"]) > 0:
        fav_timbre = pref_counts["timbre"].most_common(1)[0][0]

    fav_artist = None
    if len(pref_counts["artist"]) > 0:
        fav_artist = pref_counts["artist"].most_common(1)[0][0]

    fav_decade = None
    if len(pref_counts["decade"]) > 0:
        fav_decade = pref_counts["decade"].most_common(1)[0][0]

    scored = []

    # score each track in pool
    for tid in pool:
        meta = track_metadata[tid]

        # start with base score
        score = float(meta["base"])

        # add small feature bonuses
        if len(fav_genres) > 0:
            if meta["genre"] in fav_genres:
                score += 0.12

        if fav_timbre is not None:
            if meta["timbre"] == fav_timbre:
                score += 0.08

        if fav_decade is not None:
            if meta["decade"] == fav_decade:
                score += 0.06

        if fav_artist is not None:
            if meta["artist"] == fav_artist:
                score += 0.10

        # tiny popularity for repeating artist preference
        score += 0.03 * float(pref_counts["artist"].get(meta["artist"], 0))

        # add CF sim if exists
        score += 0.22 * float(best_cf.get(tid, 0.0))

        scored.append((tid, score))

    # sort by score desc
    scored.sort(key=lambda x: x[1], reverse=True)

    # selection part
    p_u = estimate_user_probability(song_ratings)
    exp_T = 1.0 / max(p_u, 1e-9)

    # decide exploration slots
    exploration_slots = 0
    if len(liked_tracks) > 0:
        if p_u >= 0.18:
            if topk >= 2:
                exploration_slots = 1

    anchors = topk - exploration_slots
    if anchors < 0:
        anchors = 0

    out_ids = []
    used_artist = Counter()

    # pick anchors deterministically (top scored)
    for tid, _ in scored:
        a = track_metadata[tid]["artist"]

        # enforce per-artist limit
        if used_artist[a] >= MAX_SAME_ARTIST:
            continue

        out_ids.append(tid)
        used_artist[a] += 1

        if len(out_ids) >= anchors:
            break

    # exploration sampling for remaining slots
    remaining = topk - len(out_ids)

    if remaining > 0:
        # temp depends on user patience, higher exp_T means more explore
        temp = 0.08 + 0.012 * (exp_T - 4.0)

        # clamp temp
        if temp < 0.08:
            temp = 0.08
        if temp > 0.18:
            temp = 0.18

        sample_pool = []
        sample_scores = []

        # build sample pool from top scored list
        for tid, sc in scored:
            if tid in out_ids:
                continue

            a = track_metadata[tid]["artist"]
            if used_artist[a] >= MAX_SAME_ARTIST:
                continue

            sample_pool.append(tid)
            sample_scores.append(sc)

            # keep it not too big
            if len(sample_pool) >= 240:
                break

        # do sampling if we have enough
        if len(sample_pool) > 0:
            probs = softmax(sample_scores, temp)

            picked = weighted_sample_without_replacement(
                sample_pool,
                probs,
                min(remaining, len(sample_pool))
            )

            # add picked ones
            for tid in picked:
                a = track_metadata[tid]["artist"]

                if used_artist[a] >= MAX_SAME_ARTIST:
                    continue

                out_ids.append(tid)
                used_artist[a] += 1

                if len(out_ids) >= topk:
                    break

    # fill if still not enough (deterministic)
    if len(out_ids) < topk:
        for tid, _ in scored:
            if len(out_ids) >= topk:
                break

            if tid in out_ids:
                continue

            a = track_metadata[tid]["artist"]
            if used_artist[a] >= MAX_SAME_ARTIST:
                continue

            out_ids.append(tid)
            used_artist[a] += 1

    # make final output with names
    out = []
    for tid in out_ids[:topk]:
        out.append((tid, track_metadata[tid]["name"]))

    return out


# platform entry function
def query(song_ratings: List[Dict[str, Any]], topk: int = 5) -> List[Tuple[str, str]]:
    """
    This is the entrypoint used by the platform.

    You can switch model by env:
      TD_MODEL = A or B (also 1/2)
    Default is A.
    """
    model = str(os.environ.get("TD_MODEL", "A")).strip().upper()

    # if they want model B
    if model == "B":
        return query_model_b(song_ratings, topk)
    if model == "2":
        return query_model_b(song_ratings, topk)

    # default model A
    return query_model_a(song_ratings, topk)