# src/part4.py

import os
import csv
import random
import numpy as np
import pandas as pd  # Veri üretimi için gerekli
from collections import defaultdict
import recommender   # Senin recommender.py dosyan

# --- CONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "data")
RESULTS_DIR = os.path.join(BASE_DIR, "..", "results")

# Dosya Yolları
OUTPUT_FILE_NAME = "synthetic_sessions_1200.csv"
SESSION_FILE_PATH = os.path.join(RESULTS_DIR, OUTPUT_FILE_NAME)
REPORT_FILE = os.path.join(RESULTS_DIR, "part4_simulation_results.txt")
DETAILED_LOG_FILE = os.path.join(RESULTS_DIR, "part4_detailed_logs.txt")

# Simülasyon Ayarları
NUM_SYNTHETIC_USERS = 1200
VARIATION_INTENSITY = 0.3
SIMULATION_ROUNDS = 20
INIT_HISTORY_SIZE = 5
TARGET_USERS = [200, 400, 600, 800, 1000]

# Klasör kontrolü
if not os.path.exists(RESULTS_DIR):
    try:
        os.makedirs(RESULTS_DIR)
    except OSError:
        pass

# ==========================================
# BLOCK 1: DATA GENERATION LOGIC (Eski part_test.py)
# ==========================================

def load_raw_data():
    tracks_path = os.path.join(DATA_DIR, "tracks.csv")
    ratings_path = os.path.join(DATA_DIR, "user_ratings.csv")
    
    if not os.path.exists(tracks_path) or not os.path.exists(ratings_path):
        print(f"[HATA] Kaynak veri dosyaları (tracks.csv, user_ratings.csv) bulunamadı!")
        return None, None
    return pd.read_csv(tracks_path), pd.read_csv(ratings_path)

def build_track_features(tracks_df):
    feature_map = {}
    cols = ['ab_genre_dortmund_value', 'ab_timbre_value', 'ab_mood_happy_value', 
            'ab_danceability_value']
    
    for _, row in tracks_df.iterrows():
        tid = row['track_id']
        feats = {}
        for col in cols:
            val = str(row.get(col, 'Unknown'))
            if val not in ['nan', 'Unknown']:
                feats[f"{col}:{val}"] = 1.0
        
        feats['__ARTIST__'] = str(row.get('primary_artist_name', 'Unknown'))
        feats['__POP__'] = float(row.get('track_popularity', 0)) / 100.0
        feature_map[tid] = feats
    return feature_map

def learn_user_profiles(ratings_df, track_feature_map):
    user_vectors = {}
    # Sadece 4 ve üzeri verenlerden öğren
    high_ratings = ratings_df[ratings_df['rating'] >= 4]
    
    for user_id, group in high_ratings.groupby('user_id'):
        if len(group) < 3: continue 
        
        prof_vec = defaultdict(float)
        liked_artists = defaultdict(float)
        count = 0
        
        for tid in group['song_id']:
            if tid in track_feature_map:
                feats = track_feature_map[tid]
                for k, v in feats.items():
                    if not k.startswith("__"):
                        prof_vec[k] += v
                art = feats.get('__ARTIST__')
                if art: liked_artists[art] += 1.0
                count += 1
        
        if count > 0:
            for k in prof_vec: prof_vec[k] /= count
            user_vectors[user_id] = {
                "features": prof_vec,
                "artists": liked_artists
            }
    return user_vectors

def generate_synthetic_users(seed_profiles, n):
    syn_users = []
    seeds = list(seed_profiles.keys())
    
    for i in range(n):
        seed_id = random.choice(seeds)
        seed_data = seed_profiles[seed_id]
        
        # 1. Feature Variation
        new_feats = seed_data["features"].copy()
        for k in new_feats:
            noise = 1.0 + np.random.uniform(-VARIATION_INTENSITY, VARIATION_INTENSITY)
            new_feats[k] *= noise
            
        # 2. PERSONALITY BIAS (Kişilik Atama)
        # Normal dağılım: Ortalaması -0.2 (Hafif negatif), Standart Sapma 0.6
        user_bias = np.random.normal(-0.2, 0.6)
        
        syn_users.append((f"syn_{i}_{seed_id}", {
            "features": new_feats, 
            "artists": seed_data["artists"],
            "bias": user_bias
        }))
        
    return syn_users

def score_all_tracks(syn_users, feat_map, all_tids):
    data = []
    print(f"[GEN] {len(syn_users)} bot (Farklı Kişilikli) puanlama yapıyor...")
    
    total_users = len(syn_users)
    for idx, (uid, udata) in enumerate(syn_users):
        u_feats = udata["features"]
        u_artists = udata["artists"]
        u_bias = udata["bias"]
        
        for tid in all_tids:
            t_feats = feat_map[tid]
            
            # --- SKOR FORMÜLÜ ---
            mu = 2.2 + u_bias # Base Score + Bias
            
            # Artist Match
            if t_feats.get('__ARTIST__') in u_artists:
                mu += 1.5
            
            # Feature Match
            match_score = 0.0
            for k, v in t_feats.items():
                if k in u_feats:
                    match_score += (u_feats[k] * v)
            mu += (match_score * 0.4)
            
            # Popularity Bonus
            mu += (t_feats.get('__POP__', 0) * 0.3)

            # Random Noise
            raw_score = np.random.normal(mu, 1.1)
            
            # Clamp to 1-5
            rating = int(round(raw_score))
            if rating > 5: rating = 5
            if rating < 1: rating = 1
            
            data.append([uid, tid, rating])
            
        if (idx+1) % 200 == 0:
            print(f"  > {idx+1}/{total_users} kullanıcı tamamlandı.")
            
    return pd.DataFrame(data, columns=["user_id", "song_id", "rating"])

def generate_ground_truth_if_needed():
    """Dosya yoksa üretir, varsa pas geçer."""
    if os.path.exists(SESSION_FILE_PATH):
        print(f"[INFO] '{OUTPUT_FILE_NAME}' zaten mevcut. Üretim atlanıyor.")
        return

    print(f"[INFO] '{OUTPUT_FILE_NAME}' bulunamadı. Üretim başlıyor...")
    
    tracks, ratings = load_raw_data()
    if tracks is None: return

    feat_map = build_track_features(tracks)
    profiles = learn_user_profiles(ratings, feat_map)
    
    if len(profiles) > 0:
        syn_users = generate_synthetic_users(profiles, NUM_SYNTHETIC_USERS)
        df = score_all_tracks(syn_users, feat_map, list(feat_map.keys()))
        
        # CSV Olarak Kaydet
        df.to_csv(SESSION_FILE_PATH, index=False)
        print(f"[BAŞARILI] Yeni sentetik veri oluşturuldu: {SESSION_FILE_PATH}")
        
        # Dağılım Özeti
        dist = df['rating'].value_counts(normalize=True).sort_index()
        print(f"  > Rating Dağılımı: {dist.to_dict()}")
    else:
        print("[HATA] Kullanıcı profili öğrenilemedi, veri üretilemiyor.")


# ==========================================
# BLOCK 2: SIMULATION LOGIC (Part 4)
# ==========================================

def get_conf_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    if n < 2: return np.mean(a), 0.0
    
    m = np.mean(a)
    se = np.std(a, ddof=1) / np.sqrt(n)
    z_score = 1.96 
    h = se * z_score
    return m, h

def load_full_user_data(filepath):
    users = defaultdict(dict)
    print(f"[SIM] Veriler yükleniyor: {filepath}...")
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            uid = row['user_id']
            tid = row['song_id']
            try:
                rating = int(float(row['rating']))
                users[uid][tid] = rating
            except ValueError:
                continue
    return users

def get_track_name(tid):
    if tid in recommender.track_metadata:
        return recommender.track_metadata[tid]['name']
    return "Unknown Track"

def run_simulation_step(model_func, initial_history, user_full_ratings):
    current_history = initial_history.copy()
    steps = 0
    ratings_log = []
    found_target = False

    for _ in range(SIMULATION_ROUNDS):
        steps += 1
        
        recs = model_func(current_history, topk=1)
        if not recs: break 

        rec_id = recs[0][0]
        
        # Oracle Lookup
        real_rating = user_full_ratings.get(rec_id, 1) 
        
        current_history.append({'song_id': rec_id, 'rating': real_rating})
        ratings_log.append(real_rating)

        if real_rating == 5:
            found_target = True
            break
            
    return found_target, steps, ratings_log, current_history

def write_detailed_log(f, user_idx, uid, initial_hist, hist_a, hist_b):
    f.write(f"\n{'='*60}\nUSER #{user_idx} (ID: {uid})\n{'='*60}\n")
    f.write("--- INITIAL COLD-START DATA (5 Songs) ---\n")
    for item in initial_hist:
        f.write(f"  [Rating: {item['rating']}] {get_track_name(item['song_id'])}\n")
    
    f.write("\n--- MODEL A TRACE ---\n")
    for i, item in enumerate(hist_a[len(initial_hist):], 1):
        mark = " <--- FOUND 5 STARS!" if item['rating'] == 5 else ""
        f.write(f"  Round {i:02}: [Rating: {item['rating']}] {get_track_name(item['song_id'])}{mark}\n")
        
    f.write("\n--- MODEL B TRACE ---\n")
    for i, item in enumerate(hist_b[len(initial_hist):], 1):
        mark = " <--- FOUND 5 STARS!" if item['rating'] == 5 else ""
        f.write(f"  Round {i:02}: [Rating: {item['rating']}] {get_track_name(item['song_id'])}{mark}\n")

# ==========================================
# BLOCK 3: MAIN EXECUTION
# ==========================================

def main():
    print("=== PART 4: DATA GENERATION & SIMULATION ===\n")
    
    # 1. Veri Yoksa Üret
    generate_ground_truth_if_needed()
    
    # 2. Recommender'ı Hazırla
    recommender.init_globals()
    
    # 3. Veriyi Yükle
    all_users = load_full_user_data(SESSION_FILE_PATH)
    if len(all_users) == 0: return

    # 4. Simülasyonu Başlat
    results = {
        'A': {'hit_steps': [], 'ratings': [], 'time_to_5': []},
        'B': {'hit_steps': [], 'ratings': [], 'time_to_5': []}
    }
    
    processed_count = 0
    print(f"\n[SIM] {len(all_users)} kullanıcı simüle ediliyor...")
    
    with open(DETAILED_LOG_FILE, "w", encoding="utf-8") as log_file:
        log_file.write("DETAILED SIMULATION LOGS\n")
        
        for uid, ratings_map in all_users.items():
            all_items = [{'song_id': k, 'rating': v} for k, v in ratings_map.items()]
            
            if len(all_items) < INIT_HISTORY_SIZE: continue

            seed_indices = random.sample(range(len(all_items)), INIT_HISTORY_SIZE)
            initial_history = [all_items[i] for i in seed_indices]
            
            # --- Model A ---
            hit_a, steps_a, ratings_a, history_a = run_simulation_step(
                recommender.query_model_a, initial_history, ratings_map
            )
            results['A']['ratings'].extend(ratings_a)
            results['A']['time_to_5'].append(steps_a if hit_a else SIMULATION_ROUNDS)
            results['A']['hit_steps'].append(steps_a if hit_a else 999)

            # --- Model B ---
            hit_b, steps_b, ratings_b, history_b = run_simulation_step(
                recommender.query_model_b, initial_history, ratings_map
            )
            results['B']['ratings'].extend(ratings_b)
            results['B']['time_to_5'].append(steps_b if hit_b else SIMULATION_ROUNDS)
            results['B']['hit_steps'].append(steps_b if hit_b else 999)
            
            processed_count += 1
            
            if processed_count in TARGET_USERS:
                write_detailed_log(log_file, processed_count, uid, initial_history, history_a, history_b)

            if processed_count % 200 == 0:
                print(f"  > {processed_count} kullanıcı simüle edildi.")

    # 5. Raporlama
    print("\n--- FINAL STATISTICAL REPORT ---")
    k_values = [5, 10, 20]
    report_lines = []
    header = f"{'Metric':<15} | {'Model A (Mean ± 95% CI)':<25} | {'Model B (Mean ± 95% CI)':<25}"
    report_lines.append(header)
    report_lines.append("-" * len(header))

    # Hit@K
    for k in k_values:
        ha = [1 if x <= k else 0 for x in results['A']['hit_steps']]
        hb = [1 if x <= k else 0 for x in results['B']['hit_steps']]
        ma, marga = get_conf_interval(ha)
        mb, margb = get_conf_interval(hb)
        report_lines.append(f"Hit@{k:<2} Rate    | {ma:.4f} ± {marga:.4f}       | {mb:.4f} ± {margb:.4f}")

    # Avg Rating
    ra_mean, ra_marg = get_conf_interval(results['A']['ratings'])
    rb_mean, rb_marg = get_conf_interval(results['B']['ratings'])
    report_lines.append(f"{'Avg Rating':<15} | {ra_mean:.4f} ± {ra_marg:.4f}       | {rb_mean:.4f} ± {rb_marg:.4f}")

    # Time to 5*
    ta_mean, ta_marg = get_conf_interval(results['A']['time_to_5'])
    tb_mean, tb_marg = get_conf_interval(results['B']['time_to_5'])
    report_lines.append(f"{'Time-to-5*':<15} | {ta_mean:.2f} ± {ta_marg:.2f} rounds     | {tb_mean:.2f} ± {tb_marg:.2f} rounds")

    report_text = "\n".join(report_lines)
    print(report_text)
    
    with open(REPORT_FILE, "w") as f:
        f.write(report_text)
        f.write("\n\n-- Detailed Counts --\n")
        f.write(f"Users Simulated: {processed_count}\n")
        f.write(f"Total Ratings A: {len(results['A']['ratings'])}\n")
        f.write(f"Total Ratings B: {len(results['B']['ratings'])}\n")
    
    print(f"\n[OK] Sonuçlar kaydedildi: {REPORT_FILE}")

if __name__ == "__main__":
    main()