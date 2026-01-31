# src/recommender.py
"""
CmpE 343 Project - Recommender System Implementation
----------------------------------------------------
Group: [Senin Grup Adın]

This module implements two recommendation strategies as required by Part 3.
It relies on pre-computed statistics from Part 1 and Part 2 (stored in ../results/).

Model A (Default): "Personalized Deterministic"
    - Heavily relies on Part 1 Conditional Probabilities.
    - Uses Item-Item Similarity for collaborative filtering.
    - Philosophy: Exploitation (give the user what they likely prefer).

Model B: "Adaptive Explorer"
    - Relies on Global Popularity and Part 2 User Variability.
    - Calculates user 'patience' (Tu).
    - If user is patient, it injects probabilistic exploration (Softmax).
    - Philosophy: Exploration vs. Exploitation trade-off.
"""

import os
import csv
import math
import random
from collections import defaultdict, Counter
from typing import List, Dict, Tuple, Any

# --- CONFIGURATION & PATHS ---
BASE_DIR = os.path.dirname(__file__)
RESULTS_DIR = os.path.join(BASE_DIR, "..", "results")

# Pre-computed CSV files from Part 1 and Part 2
PART1_PRIORS_PATH = os.path.join(RESULTS_DIR, "part1_feature_priors.csv")
PART1_META_PATH = os.path.join(RESULTS_DIR, "part1_track_meta.csv")
PART2_PRIORS_PATH = os.path.join(RESULTS_DIR, "part2_priors.csv")
ITEM_SIM_PATH = os.path.join(RESULTS_DIR, "part3_item_sim.csv")

# Hyperparameters
LIKE_THRESHOLD = 4
DISLIKE_THRESHOLD = 2
MAX_PER_ARTIST = 2  # Diversity constraint

# --- GLOBAL MEMORY ---
_loaded = False
track_metadata: Dict[str, Dict[str, Any]] = {}
global_ranking: List[str] = []
priors_by_feature = defaultdict(dict)
item_similarity = defaultdict(list)
tracks_by_artist = defaultdict(list)
tracks_by_genre = defaultdict(list)
tracks_by_timbre = defaultdict(list)
tracks_by_decade = defaultdict(list)
tracks_by_popbin = defaultdict(list)

# Default Part 2 priors (updated from CSV if available)
alpha_prior = 4.33
beta_prior = 6.93

def _safe_str(x: Any) -> str:
    return str(x).strip() if x is not None else ""

def parse_track_id(row: Dict[str, Any]) -> str:
    """Extracts track ID robustly from input dictionaries."""
    for key in ["spotify_id", "song_id", "track_id", "id"]:
        if key in row and row[key] is not None:
            s = _safe_str(row[key])
            if s != "": return s
    return ""

def _read_csv_dicts(path: str) -> List[Dict[str, str]]:
    rows = []
    if not os.path.exists(path):
        return rows
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    return rows

def init_globals():
    """
    Loads all pre-computed data into memory once.
    This ensures the 'query' function runs within the 8-second limit.
    """
    global _loaded, alpha_prior, beta_prior
    if _loaded: return

    # 1. Load Track Metadata (Part 1 Base Scores)
    meta_rows = _read_csv_dicts(PART1_META_PATH)
    for r in meta_rows:
        tid = _safe_str(r.get("track_id", ""))
        if not tid: continue
        
        # Parse fields
        track_metadata[tid] = {
            "name": _safe_str(r.get("track_name", "")),
            "artist": _safe_str(r.get("primary_artist_name", "Unknown")),
            "genre": _safe_str(r.get("ab_genre_dortmund_value", "Unknown")),
            "timbre": _safe_str(r.get("ab_timbre_value", "Unknown")),
            "mood": _safe_str(r.get("ab_mood_happy_value", "Unknown")),
            "decade": _safe_str(r.get("release_decade", "Unknown")),
            "popbin": _safe_str(r.get("pop_bin", "Unknown")),
            "base": float(r.get("base", 0.0)),
            "rating_count": int(float(r.get("rating_count", 0)))
        }

    # Create indices for fast lookup
    global_ranking.extend(sorted(track_metadata.keys(), key=lambda t: track_metadata[t]["base"], reverse=True))
    for tid in global_ranking:
        m = track_metadata[tid]
        tracks_by_artist[m["artist"]].append(tid)
        tracks_by_genre[m["genre"]].append(tid)
        tracks_by_timbre[m["timbre"]].append(tid)
        tracks_by_decade[m["decade"]].append(tid)
        tracks_by_popbin[m["popbin"]].append(tid)

    # 2. Load Part 1 Feature Priors
    prior_rows = _read_csv_dicts(PART1_PRIORS_PATH)
    for r in prior_rows:
        feat = _safe_str(r.get("feature", ""))
        val = _safe_str(r.get("value", ""))
        if feat and val:
            priors_by_feature[feat][val] = float(r.get("p_5", 0.20))

    # 3. Load Part 2 Beta Parameters
    p2_rows = _read_csv_dicts(PART2_PRIORS_PATH)
    if p2_rows:
        try:
            alpha_prior = float(p2_rows[0].get("alpha", alpha_prior))
            beta_prior = float(p2_rows[0].get("beta", beta_prior))
        except: pass

    # 4. Load Item-Item Similarity
    sim_rows = _read_csv_dicts(ITEM_SIM_PATH)
    tmp_sim = defaultdict(list)
    for r in sim_rows:
        a = _safe_str(r.get("track_id", ""))
        b = _safe_str(r.get("neighbor_id", ""))
        if a and b:
            tmp_sim[a].append((b, float(r.get("sim", 0.0))))
    
    for a in tmp_sim:
        tmp_sim[a].sort(key=lambda x: x[1], reverse=True)
        item_similarity[a] = tmp_sim[a]

    _loaded = True

# --- SHARED HELPERS ---

def build_user_profile(song_ratings):
    """Analyzes user input to find liked tracks and preferred features."""
    seen_tracks = set()
    liked_tracks = []
    disliked_artists = set()
    pref_counts = {
        "artist": Counter(), "genre": Counter(), "timbre": Counter(),
        "mood": Counter(), "decade": Counter(), "popbin": Counter()
    }

    for row in song_ratings:
        tid = parse_track_id(row)
        if not tid or tid not in track_metadata: continue
        seen_tracks.add(tid)
        
        try: rating = int(float(row.get("rating", 0)))
        except: continue

        meta = track_metadata[tid]
        if rating >= LIKE_THRESHOLD:
            liked_tracks.append(tid)
            pref_counts["artist"][meta["artist"]] += 1
            pref_counts["genre"][meta["genre"]] += 1
            pref_counts["timbre"][meta["timbre"]] += 1
            pref_counts["mood"][meta["mood"]] += 1
            pref_counts["decade"][meta["decade"]] += 1
            pref_counts["popbin"][meta["popbin"]] += 1
        elif rating <= DISLIKE_THRESHOLD:
            disliked_artists.add(meta["artist"])
            
    return liked_tracks, seen_tracks, disliked_artists, pref_counts

# --- MODEL A IMPLEMENTATION ---

def session_prior_probs(song_ratings, meta_key):
    """Calculates P(5*|Feature) specifically for the current session."""
    total = Counter()
    five = Counter()
    alpha = 1.0 # Smoothing

    for row in song_ratings:
        tid = parse_track_id(row)
        if tid not in track_metadata: continue
        try: r = int(float(row.get("rating", 0)))
        except: continue
        
        val = track_metadata[tid].get(meta_key, "Unknown")
        total[val] += 1
        if r == 5: five[val] += 1
        
    out = {}
    for v in total:
        out[v] = (float(five[v]) + alpha) / (float(total[v]) + 2.0 * alpha)
    return out

def query_model_a(song_ratings, topk):
    """
    Model A: Personalized Deterministic.
    Uses a weighted score of: Base Quality + Session Probabilities + Global Probabilities + CF
    """
    init_globals()
    liked_tracks, seen, disliked_artists, prefs = build_user_profile(song_ratings)
    
    # Cold start fallback
    if not liked_tracks:
        return query_model_b(song_ratings, topk)

    # 1. Candidate Generation
    candidates = defaultdict(float)
    
    def add(tid, w):
        if tid in seen or tid not in track_metadata: return
        if track_metadata[tid]["artist"] in disliked_artists: return
        candidates[tid] += w

    # A) Collaborative Filtering (Neighbors of liked songs)
    for i, lid in enumerate(liked_tracks):
        # Recent likes get slightly more weight
        w_recency = 1.0 + 0.5 * (i / len(liked_tracks))
        for nb, sim in item_similarity.get(lid, []):
            add(nb, sim * 4.0 * w_recency)

    # B) Feature Expansion (Top genres/artists)
    for g, _ in prefs["genre"].most_common(2):
        for tid in tracks_by_genre.get(g, [])[:150]: add(tid, 1.2)
    for a, _ in prefs["artist"].most_common(2):
        for tid in tracks_by_artist.get(a, [])[:100]: add(tid, 1.8)

    # C) Global Backfill
    for tid in global_ranking[:500]: add(tid, 0.2)

    # 2. Scoring with Conditional Probabilities (Part 1 Insight)
    # Blend global P(5|F) with session P(5|F)
    mapping = [("ab_genre_dortmund_value", "genre"), ("primary_artist_name", "artist")]
    blended_probs = defaultdict(dict)
    
    for feat_key, meta_key in mapping:
        sess_probs = session_prior_probs(song_ratings, meta_key)
        glob_probs = priors_by_feature.get(feat_key, {})
        for val in set(list(sess_probs.keys()) + list(glob_probs.keys())):
            # Weighted average: prioritize session if data exists
            p_s = sess_probs.get(val, 0.2)
            p_g = glob_probs.get(val, 0.2)
            blended_probs[meta_key][val] = 0.7 * p_s + 0.3 * p_g

    final_scores = []
    for tid, raw_score in candidates.items():
        m = track_metadata[tid]
        
        # Probabilistic Boost
        p_boost = 0.0
        p_boost += blended_probs["genre"].get(m["genre"], 0)
        p_boost += blended_probs["artist"].get(m["artist"], 0)
        
        final_score = raw_score + (3.0 * m["base"]) + (2.0 * p_boost)
        final_scores.append((tid, final_score))

    # 3. Selection & Formatting
    final_scores.sort(key=lambda x: x[1], reverse=True)
    
    result = []
    used_artists = Counter()
    for tid, _ in final_scores:
        art = track_metadata[tid]["artist"]
        if used_artists[art] >= MAX_PER_ARTIST: continue
        
        result.append((tid, track_metadata[tid]["name"]))
        used_artists[art] += 1
        if len(result) >= topk: break
        
    return result

# --- MODEL B IMPLEMENTATION ---

def estimate_user_patience(song_ratings):
    """Estimates user probability p_u using Beta-Geometric update (Part 2)."""
    five_count = sum(1 for r in song_ratings if int(float(r.get("rating", 0))) == 5)
    total_count = len(song_ratings)
    # Posterior mean of Beta(alpha + hits, beta + misses)
    # Here, simply updating p_u expectation
    return (alpha_prior + five_count) / (alpha_prior + beta_prior + total_count)

def softmax(scores, temp=0.1):
    mx = max(scores)
    exps = [math.exp((s - mx) / temp) for s in scores]
    tot = sum(exps)
    return [e/tot for e in exps]

def query_model_b(song_ratings, topk):
    """
    Model B: Adaptive Explorer.
    Uses Global Popularity as a baseline.
    Checks User Patience (Part 2). If patient, performs probabilistic exploration.
    """
    init_globals()
    liked, seen, disliked_artists, prefs = build_user_profile(song_ratings)
    
    # 1. Pool Creation (Global + Some Content Match)
    pool = set(global_ranking[:800]) # Top global
    
    # Add some tracks from top genre to ensure relevance
    top_genre = prefs["genre"].most_common(1)
    if top_genre:
        g = top_genre[0][0]
        for tid in tracks_by_genre.get(g, [])[:200]: pool.add(tid)

    # 2. Base Scoring
    scored_pool = []
    for tid in pool:
        if tid in seen: continue
        m = track_metadata[tid]
        if m["artist"] in disliked_artists: continue
        
        # Base score from Part 1
        score = m["base"]
        # Slight bump for genre match
        if top_genre and m["genre"] == top_genre[0][0]: score += 0.15
            
        scored_pool.append((tid, score))
        
    scored_pool.sort(key=lambda x: x[1], reverse=True)
    
    # 3. Adaptive Strategy (Part 2 Insight)
    p_u = estimate_user_patience(song_ratings)
    expected_Tu = 1.0 / p_u # Expected rounds to find a favorite
    
    # If user expects many rounds (patient) or we have few recs requested, explore.
    is_patient = (p_u < 0.25) 
    
    result = []
    used_artists = Counter()
    
    # Strategy: Anchors (Deterministic) + Explorers (Probabilistic)
    num_anchors = topk - 1 if (is_patient and topk > 1) else topk
    
    # A) Fill Anchors (Safe bets)
    for tid, sc in scored_pool:
        if len(result) >= num_anchors: break
        art = track_metadata[tid]["artist"]
        if used_artists[art] >= MAX_PER_ARTIST: continue
        result.append(tid)
        used_artists[art] += 1
        
    # B) Fill Explorer Slot (Softmax Sampling)
    if len(result) < topk:
        # Filter pool for candidates not yet picked
        exploration_candidates = []
        exploration_scores = []
        
        for tid, sc in scored_pool[:150]: # Look at top 150 candidates
            if tid not in result:
                art = track_metadata[tid]["artist"]
                if used_artists[art] < MAX_PER_ARTIST:
                    exploration_candidates.append(tid)
                    exploration_scores.append(sc)
        
        if exploration_candidates:
            # Temperature: Higher Tu (lower p_u) -> Higher Temp (More random)
            temp = 0.05 + (0.1 / (p_u * 10 + 0.1)) 
            probs = softmax(exploration_scores, temp=temp)
            
            # Weighted choice
            pick = random.choices(exploration_candidates, weights=probs, k=1)[0]
            result.append(pick)
    
    # Format output
    final_output = []
    for tid in result:
        final_output.append((tid, track_metadata[tid]["name"]))
        
    # Final cleanup backfill if needed
    if len(final_output) < topk:
        for tid in global_ranking:
            if len(final_output) >= topk: break
            if tid not in seen and tid not in [x[0] for x in final_output]:
                final_output.append((tid, track_metadata[tid]["name"]))
                
    return final_output[:topk]

# --- MAIN ENTRY POINT ---

def query(song_ratings: List[Dict[str, Any]], topk: int = 5) -> List[Tuple[str, str]]:
    """
    API Function called by the platform.
    
    Arguments:
    song_ratings -- List of dictionaries containing user history.
                    e.g. [{'song_id': '...', 'rating': 5}, ...]
    topk         -- Number of recommendations required.
    
    Returns:
    List of tuples (track_id, track_name).
    """
    # Environment variable can switch model for testing/comparison
    # Default is 'A' (Personalized)
    model_choice = os.environ.get("TD_MODEL", "A").upper()
    
    if model_choice == "B":
        return query_model_b(song_ratings, topk)
    else:
        return query_model_a(song_ratings, topk)

# For testing locally
if __name__ == "__main__":
    # Mock data
    mock_history = [
        {"song_id": "6UelLqGlWMcVH1E5c4H7lY", "rating": 5}, # Watermelon Sugar -  Harry Styles
        {"song_id": "3GCdLUSnKSMJhs4Tj6CV3s", "rating": 5}  # All The Stars - Kendrick Lamar
    ]
    print("Testing Model A...")
    print(query_model_a(mock_history, 5))
    print("Testing Model B...")
    print(query_model_b(mock_history, 5))