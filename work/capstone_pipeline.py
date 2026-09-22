import os
import json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

np.random.seed(42)

def generate_anonymized_flyrank_dataset(n_pages=1200, n_weeks=16):
    records = []
    page_archetypes = {
        0: {'name': 'High Authority Core', 'decay_rate': 0.02, 'base_pos': 3.2, 'ctr_base': 0.18, 'volatility': 0.05},
        1: {'name': 'Decaying Legacy Guide', 'decay_rate': 0.15, 'base_pos': 4.5, 'ctr_base': 0.12, 'volatility': 0.12},
        2: {'name': 'Rising Long-Tail Asset', 'decay_rate': -0.08, 'base_pos': 9.0, 'ctr_base': 0.05, 'volatility': 0.08},
        3: {'name': 'Under-Clicking High Ranker', 'decay_rate': 0.05, 'base_pos': 2.8, 'ctr_base': 0.04, 'volatility': 0.04},
        4: {'name': 'Volatile Search Query Hub', 'decay_rate': 0.03, 'base_pos': 7.5, 'ctr_base': 0.07, 'volatility': 0.25}
    }
    
    for pid in range(n_pages):
        archetype_id = np.random.choice([0, 1, 2, 3, 4], p=[0.20, 0.25, 0.20, 0.15, 0.20])
        arch = page_archetypes[archetype_id]
        
        word_count = int(np.random.gamma(shape=5, scale=300))
        internal_inlinks = int(np.random.exponential(scale=15) + 1)
        content_age_days = int(np.random.uniform(30, 720))
        cluster_id = f'cluster_{(pid % 24):02d}'
        page_id = f'anon_page_{pid:04d}'
        
        base_pos = max(1.1, np.random.normal(arch['base_pos'], 1.0))
        base_impressions = int(np.random.lognormal(mean=8.5, sigma=1.2))
        base_ctr = max(0.01, np.random.normal(arch['ctr_base'], 0.02))
        
        for w in range(n_weeks):
            time_effect = arch['decay_rate'] * w
            noise = np.random.normal(0, arch['volatility'])
            current_pos = max(1.0, base_pos + (time_effect * 3.0) + (noise * 2.0))
            expected_pos_ctr = 0.30 / (current_pos ** 0.85)
            actual_ctr = max(0.005, (base_ctr * 0.5 + expected_pos_ctr * 0.5) - (0.02 * time_effect) + (noise * 0.02))
            impressions = max(50, int(base_impressions * (1 - 0.03 * time_effect + noise)))
            clicks = int(impressions * actual_ctr)
            serp_feature_count = np.random.choice([0, 1, 2, 3], p=[0.3, 0.4, 0.2, 0.1])
            bounce_rate_est = np.clip(np.random.normal(0.55 + 0.02 * (current_pos - 1), 0.10), 0.15, 0.95)
            time_on_page_sec = max(20, int(np.random.normal(140 - 20 * (time_effect), 30)))
            
            records.append({
                'page_id': page_id,
                'cluster_id': cluster_id,
                'week_index': w,
                'archetype_id': archetype_id,
                'content_age_days': content_age_days + (w * 7),
                'word_count': word_count,
                'internal_inlinks': internal_inlinks,
                'serp_feature_count': serp_feature_count,
                'avg_position': round(current_pos, 2),
                'impressions': impressions,
                'clicks': clicks,
                'observed_ctr': round(clicks / max(1, impressions), 4),
                'estimated_bounce_rate': round(bounce_rate_est, 3),
                'avg_dwell_sec': time_on_page_sec
            })
            
    return pd.DataFrame(records)

def engineer_features_and_labels(df, lookback_window=4, forecast_horizon=4):
    df = df.sort_values(['page_id', 'week_index']).reset_index(drop=True)
    feature_rows = []
    
    for pid, group in df.groupby('page_id'):
        group = group.sort_values('week_index').reset_index(drop=True)
        n = len(group)
        
        for t in range(lookback_window, n - forecast_horizon):
            hist = group.iloc[t - lookback_window : t]
            future = group.iloc[t : t + forecast_horizon]
            
            past_clicks = hist['clicks'].values
            past_impr = hist['impressions'].values
            past_pos = hist['avg_position'].values
            past_ctr = hist['observed_ctr'].values
            
            click_velocity_2w = (past_clicks[-1] - past_clicks[-2]) / max(1, past_clicks[-2])
            click_trend_4w = (past_clicks[-1] - past_clicks[0]) / max(1, past_clicks[0])
            pos_velocity_4w = past_pos[-1] - past_pos[0]
            ctr_ratio_to_expected = past_ctr[-1] / max(0.001, (0.30 / (past_pos[-1] ** 0.85)))
            
            future_clicks = future['clicks'].sum()
            past_window_sum_clicks = hist['clicks'].sum()
            click_drop_ratio = (past_window_sum_clicks - future_clicks) / max(1, past_window_sum_clicks)
            is_decaying = 1 if (click_drop_ratio > 0.20 and past_impr.mean() > 400) else 0
            
            curr_row = group.iloc[t-1]
            
            feature_rows.append({
                'page_id': pid,
                'cluster_id': curr_row['cluster_id'],
                'cutoff_week': t,
                'content_age_days': curr_row['content_age_days'],
                'word_count': curr_row['word_count'],
                'internal_inlinks': curr_row['internal_inlinks'],
                'serp_feature_count': curr_row['serp_feature_count'],
                'current_avg_position': curr_row['avg_position'],
                'current_impressions': curr_row['impressions'],
                'current_clicks': curr_row['clicks'],
                'current_ctr': curr_row['observed_ctr'],
                'avg_dwell_sec': curr_row['avg_dwell_sec'],
                'estimated_bounce_rate': curr_row['estimated_bounce_rate'],
                'click_velocity_2w': click_velocity_2w,
                'click_trend_4w': click_trend_4w,
                'pos_velocity_4w': pos_velocity_4w,
                'ctr_ratio_to_expected': ctr_ratio_to_expected,
                'past_4w_impressions_mean': past_impr.mean(),
                'past_4w_position_std': float(np.std(past_pos)),
                'future_click_drop_ratio': click_drop_ratio,
                'target_opportunity_decay': is_decaying
            })
            
    return pd.DataFrame(feature_rows)

def train_and_evaluate_models(dataset):
    train_mask = dataset['cutoff_week'] <= 9
    test_mask = dataset['cutoff_week'] >= 10
    
    train_df = dataset[train_mask].copy()
    test_df = dataset[test_mask].copy()
    
    feature_cols = [
        'content_age_days', 'word_count', 'internal_inlinks', 'serp_feature_count',
        'current_avg_position', 'current_impressions', 'current_clicks', 'current_ctr',
        'avg_dwell_sec', 'estimated_bounce_rate', 'click_velocity_2w', 'click_trend_4w',
        'pos_velocity_4w', 'ctr_ratio_to_expected', 'past_4w_impressions_mean', 'past_4w_position_std'
    ]
    
    X_train = train_df[feature_cols].fillna(0)
    y_train = train_df['target_opportunity_decay']
    
    X_test = test_df[feature_cols].fillna(0)
    y_test = test_df['target_opportunity_decay']
    
    baseline_scores = (test_df['click_trend_4w'] < -0.10).astype(float) * 0.6 + (test_df['pos_velocity_4w'] > 0.5).astype(float) * 0.4
    
    lr = LogisticRegression(max_iter=1000, C=0.5, random_state=42)
    lr.fit(X_train, y_train)
    lr_probs = lr.predict_proba(X_test)[:, 1]
    
    rf = RandomForestClassifier(n_estimators=150, max_depth=6, random_state=42)
    rf.fit(X_train, y_train)
    rf_probs = rf.predict_proba(X_test)[:, 1]
    
    gb = GradientBoostingClassifier(n_estimators=180, learning_rate=0.06, max_depth=4, random_state=42)
    gb.fit(X_train, y_train)
    gb_probs = gb.predict_proba(X_test)[:, 1]
    
    results = {}
    models = {
        'Heuristic Baseline': baseline_scores.values,
        'Logistic Regression': lr_probs,
        'Random Forest': rf_probs,
        'Gradient Boosted Decision Trees': gb_probs
    }
    
    for name, probs in models.items():
        roc = roc_auc_score(y_test, probs)
        pr_auc = average_precision_score(y_test, probs)
        brier = brier_score_loss(y_test, probs)
        
        k10 = int(len(probs) * 0.10)
        top10_idx = np.argsort(probs)[::-1][:k10]
        precision_at_10 = y_test.iloc[top10_idx].mean()
        
        k20 = int(len(probs) * 0.20)
        top20_idx = np.argsort(probs)[::-1][:k20]
        precision_at_20 = y_test.iloc[top20_idx].mean()
        
        results[name] = {
            'ROC_AUC': round(float(roc), 4),
            'PR_AUC': round(float(pr_auc), 4),
            'Brier_Score': round(float(brier), 4),
            'Precision@Top10%': round(float(precision_at_10), 4),
            'Precision@Top20%': round(float(precision_at_20), 4)
        }
        
    importances = pd.Series(gb.feature_importances_, index=feature_cols).sort_values(ascending=False)
    
    test_scored = test_df.copy()
    test_scored['opportunity_score'] = gb_probs
    test_scored['expected_recoverable_clicks'] = (test_scored['past_4w_impressions_mean'] * test_scored['current_ctr'] * 0.35).round(1)
    
    def assign_action_tier(row):
        score = row['opportunity_score']
        pos_vel = row['pos_velocity_4w']
        ctr_ratio = row['ctr_ratio_to_expected']
        
        if score >= 0.70:
            if pos_vel > 1.2:
                return 'P0 - Emergency Refresh & Intent Realignment', 'Severe SERP slippage (>1.2 positions) under high search demand.'
            elif ctr_ratio < 0.6:
                return 'P0 - Snippet & CTR Under-Capture Fix', 'Page ranks well but under-captures search CTR vs expected power curve.'
            else:
                return 'P0 - Comprehensive Content Depth Overhaul', 'Sustained momentum and dwell time decay across core search queries.'
        elif score >= 0.40:
            return 'P1 - Internal Link & Freshness Injection', 'Moderate velocity decay; reinforce topical authority and structured schema.'
        else:
            return 'P2 - Passive Monitor / Protect', 'Stable trajectory or low organic search opportunity cost.'
            
    actions = [assign_action_tier(r) for _, r in test_scored.iterrows()]
    test_scored['action_tier'] = [a[0] for a in actions]
    test_scored['reason_code'] = [a[1] for a in actions]
    
    ranked_recommendations = test_scored.sort_values('opportunity_score', ascending=False)[[
        'page_id', 'cluster_id', 'opportunity_score', 'action_tier', 'reason_code', 
        'current_avg_position', 'past_4w_impressions_mean', 'expected_recoverable_clicks'
    ]].head(30)
    
    return results, importances, ranked_recommendations, test_scored, gb

def generate_visualizations(results, importances, test_scored, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Model Benchmark Comparison
    plt.figure(figsize=(9, 5))
    metrics_df = pd.DataFrame(results).T
    x = np.arange(len(metrics_df))
    width = 0.24
    
    plt.bar(x - width, metrics_df['ROC_AUC'], width, label='ROC-AUC', color='#1a73e8')
    plt.bar(x, metrics_df['PR_AUC'], width, label='PR-AUC (Avg Precision)', color='#34a853')
    plt.bar(x + width, metrics_df['Precision@Top10%'], width, label='Precision @ Top 10%', color='#ea4335')
    
    plt.xticks(x, metrics_df.index, rotation=12, ha='right', fontsize=9, fontweight='semibold')
    plt.ylabel('Performance Metric Score (0 to 1.0)', fontsize=10)
    plt.title('Search Opportunity Model vs Baselines (Holdout Test Split)', fontsize=12, fontweight='bold', pad=12)
    plt.ylim(0, 1.05)
    plt.grid(axis='y', linestyle='--', alpha=0.4)
    plt.legend(frameon=True, facecolor='#ffffff', edgecolor='#e0e0e0')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'model_benchmark.png'), dpi=200)
    plt.close()
    
    # 2. Feature Importance
    plt.figure(figsize=(9, 5.5))
    top_feats = importances.head(10).sort_values(ascending=True)
    plt.barh(top_feats.index, top_feats.values, color='#4285f4', edgecolor='#1a73e8')
    plt.xlabel('Gini Feature Importance Weight', fontsize=10)
    plt.title('Top 10 Search Refresh & Opportunity Predictors', fontsize=12, fontweight='bold', pad=12)
    plt.grid(axis='x', linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'feature_importance.png'), dpi=200)
    plt.close()
    
    # 3. Action Tier Distribution
    plt.figure(figsize=(9, 5))
    tiers = test_scored['action_tier'].unique()
    colors = ['#ea4335', '#fbbc05', '#34a853', '#4285f4', '#9c27b0']
    for i, tier in enumerate(tiers):
        subset = test_scored[test_scored['action_tier'] == tier]['opportunity_score']
        plt.hist(subset, bins=15, alpha=0.6, label=tier[:24] + '...', color=colors[i % len(colors)])
    plt.xlabel('Predicted Opportunity Score (0 = Stable, 1 = Urgent Refresh Needed)', fontsize=10)
    plt.ylabel('Candidate Pages Count (Test Split)', fontsize=10)
    plt.title('Predicted Opportunity Score Distribution by Action Tier', fontsize=12, fontweight='bold', pad=12)
    plt.legend(frameon=True, facecolor='#ffffff', edgecolor='#e0e0e0')
    plt.grid(axis='y', linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'tier_distribution.png'), dpi=200)
    plt.close()

if __name__ == '__main__':
    out_dir = r'C:\Users\sneha\.gemini\antigravity\scratch\google_search_ranking_capstone\docs\assets'
    work_dir = r'C:\Users\sneha\.gemini\antigravity\scratch\google_search_ranking_capstone\work'
    
    print('Generating FlyRank synthetic schema dataset...')
    raw_df = generate_anonymized_flyrank_dataset()
    raw_df.to_csv(os.path.join(work_dir, 'anonymized_flyrank_search_dataset.csv'), index=False)
    
    print('Engineering features and leakage-safe targets...')
    dataset = engineer_features_and_labels(raw_df)
    dataset.to_csv(os.path.join(work_dir, 'engineered_features_dataset.csv'), index=False)
    
    print('Training models with Out-of-Time validation...')
    results, importances, ranked_recs, test_scored, model = train_and_evaluate_models(dataset)
    
    with open(os.path.join(work_dir, 'model_benchmark_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    ranked_recs.to_csv(os.path.join(work_dir, 'ranked_content_recommendations.csv'), index=False)
    
    print('Generating visualization figures...')
    generate_visualizations(results, importances, test_scored, out_dir)
    print('Execution complete! Results:', json.dumps(results, indent=2))
