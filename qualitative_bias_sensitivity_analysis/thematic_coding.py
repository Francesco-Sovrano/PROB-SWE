import re
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

themes_keywords = {
    'performance': ['availability', 'efficient', 'latency', 'optimize', 'load', 'scalability', 'speed', 'optimization', 'performance', 'improve', 'quick', 'efficiency', 'fast', 'processing', 'asynchronous', 'bandwidth', 'stability', 'downtime', 'productivity', 'throughput', 'compression', 'reliability', 'multiplexing', 'scalable', 'slow', 'benchmark', 'grow', 'better', 'exponential'],
    'security': ['protection', 'vulnerabilities', 'intrusion', 'risk', 'privacy', 'encryption', 'access', 'robustness', 'vulnerability', 'auth', 'cve', 'audit', 'attack', 'security', 'compliance', 'secure', 'authorization', 'threat', 'gdpr', 'logging', 'monitoring'],
    'team': ['collaboration', 'expertise', 'agility', 'skill', 'staff', 'training', 'meetings', 'management', 'manageability', 'refactor', 'team', 'onboarding', 'staffing', 'maintainability', 'mentorship', 'organization', 'knowledge', 'capability', 'communicate'],
    'customer': ['communicat', 'community', 'functionality', 'company', 'stakeholder', 'feature', 'customer', 'satisfaction', 'user experience', 'usability', 'feedback', 'client', 'reputation', 'industry', 'market', 'user', 'maintenance', 'business', 'retention', 'commerce'],
    'cost': [
        'overhead', 'cost', 'sacrificing',
        'investment',  # already present
        'resource'     # “resource allocation,” “resource limits”
    ],
    'timeline': ['delivery', 'lag', 'time', 'late', 'immediately', 'schedule', 'scheduling', 'wait', 'longer', 'roadmap', 'delay', 'deadline', 'milestone', 'now', 'urgent', 'quick', 'crucial', 'cut-off', 'prompt', 'immediate', 'promptly', 'fast', 'quick', 'slow', 'term'],
    'trendiness': ['state-of-the-art', 'prototype', 'community', 'cutting-edge', 'latest', 'adoption', 'modern', 'trend', 'emerging', 'common', 'new library', 'beta', 'innovative', 'up-to-date'],
    'maintainability': ['deprecated', 'technical debt', 'manageable', 'maintained', 'maintain', 'manageability', 'refactor', 'maintainable', 'maintenance', 'refactoring', 'maintainability', 'maintaining', 'logging', 'deployment']
,
    'reliability': ['failure', 'vulnerabilities', 'instability', 'consistency', 'robustness', 'redundancy', 'fail-safe', 'uptime', 'resilience', 'rollback', 'robust', 'maintainability', 'reliability', 'fault-tolerant'],
    'scalability': ['load balancing', 'scale', 'shard', 'elasticity', 'limits', 'scalability', 'optimized', 'performance', 'complexity', 'optimizations', 'manageability', 'throttling', 'stability', 'maintainability', 'scalable', 'containerization', 'scaling'],
    'quality': [
      'test',         # unit, integration, smoke, etc.
      'testing',      # verb/noun form
      'debugging',
      'verification',
      'validation',
      'qa',
    ],
    'usability': ['customization', 'intuitive', 'ux', 'design', 'documentation', 'ui', 'ergonomic', 'experience', 'interface', 'frustrating'],
    'compliance': ['policies', 'security', 'audit', 'compliance', 'audits', 'guidelines', 'oversight', 'accountability', 'policy', 'documentation', 'regulation', 'practices', 'standard', 'hipaa', 'gdpr'],
    'effort': ['learning', 'challenging', 'straightforward', 'ramp-up', 'effort', 'implementing', 'making', 'optimized', 'training', 'hand-roll', 'simplicity', 'improving', 'easier', 'improves', 'complexity', 'optimizations', 'simpler', 'staffing', 'improvements', 'manual'],
}

themes = [
    f"software {k}"
    for k,v in themes_keywords.items()
]

# Load the CSV file
df = pd.read_csv('bias_sample_analysis.csv')

# cols = ['decision_explanation_without_bias','decision_explanation_with_bias']
# words = set()
# for col in cols:
#     df[col].dropna().astype(str).apply(lambda x: words.update(re.findall(r'\b\w{4,}\b', x.lower())))

# words = sorted(words)

# # Load pre-trained embedding model
# model = SentenceTransformer('all-MiniLM-L6-v2')

# # Embed theme names and words
# theme_embs = model.encode(themes, convert_to_tensor=False)
# word_embs = model.encode(words, convert_to_tensor=False)

# # Compute similarity matrix
# sim_matrix = cosine_similarity(word_embs, theme_embs)

# # Build mapping with threshold
# threshold = 0.35
# for i, theme in enumerate(themes_keywords.keys()):
#     similar_words = [words[j] for j in range(len(words)) if sim_matrix[j, i] >= threshold]
#     print(theme, json.dumps(similar_words, indent=4))
#     themes_keywords[theme] = list(set(themes_keywords[theme]+similar_words))
# assert False

# print('themes_keywords:', json.dumps(themes_keywords, indent=4))

# Function to assign themes based on keywords
def assign_themes(text):
    if not isinstance(text, str):
        return []
    t = text.lower()
    return [theme for theme, kws in themes_keywords.items() if any(k in t for k in kws)]

# Load and annotate
df = pd.read_csv('bias_sample_analysis.csv')
for suffix in ['without_bias', 'with_bias']:
    col = f"decision_explanation_{suffix}"
    df[f"themes_{suffix}"] = df[col].apply(assign_themes)

# Save thematic coding
out = df[[
    'decision_explanation_without_bias',
    'decision_explanation_with_bias',
    'themes_without_bias',
    'themes_with_bias'
]]
out.to_csv('thematic_coding.csv', index=False)
print("Saved thematic_coding.csv")

# Prepare counts for grouped bar chart
counts = {
    theme: (
        df['themes_without_bias'].explode().value_counts().get(theme, 0),
        df['themes_with_bias'].explode().value_counts().get(theme, 0)
    ) for theme in sorted(themes_keywords)
}
labels, without_vals, with_vals = zip(*[(t, v[0], v[1]) for t, v in counts.items()])
x = np.arange(len(labels))
width = 0.35

# get two distinct, color-blind–friendly colors from the 'tab10' colormap
cmap   = plt.get_cmap('tab10')
colors = [cmap(0), cmap(1)]

fig, ax = plt.subplots(figsize=(8, 4))
ax.bar(x - width/2, without_vals, width,
       label='Without Bias',
       color=colors[0])
ax.bar(x + width/2, with_vals, width,
       label='With Bias',
       color=colors[1])
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=45, ha='right')
ax.set_ylabel('Count')
# ax.set_title('Theme Frequencies')
ax.legend()
ax.grid(axis='y', linestyle='--', alpha=0.6)
for i, (w, b) in enumerate(zip(without_vals, with_vals)):
    ax.text(i - width/2, w + 0.1, str(w), ha='center', va='bottom')
    ax.text(i + width/2, b + 0.1, str(b), ha='center', va='bottom')
plt.tight_layout()
plt.savefig("theme_frequency.pdf", dpi=300, bbox_inches='tight')
plt.show()

# Heatmap of Theme Differences
melted = df.melt(id_vars='bias_name',
                 value_vars=['themes_without_bias', 'themes_with_bias'],
                 var_name='code_type', value_name='themes')
melted = melted.explode('themes').dropna()
map_type = {'themes_without_bias': 'without', 'themes_with_bias': 'with'}
melted['type'] = melted['code_type'].map(map_type)

# Pivot into counts and compute difference
grouped = melted.groupby(['themes', 'bias_name', 'type']).size().unstack(fill_value=0)
diff_df = (grouped.get('with', pd.Series()) - grouped.get('without', pd.Series())).unstack(fill_value=0)

# Plot heatmap
plt.figure(figsize=(8, 5))
sns.heatmap(diff_df.fillna(0), annot=True, cmap='icefire', cbar_kws={'label': 'Δ Count'})
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.xlabel('')
plt.ylabel('Theme')
# plt.title('Δ Theme Counts (With - Without)')
plt.tight_layout()
plt.savefig("heatmap_theme_count_delta_by_bias.pdf", dpi=300, bbox_inches='tight')
plt.show()
