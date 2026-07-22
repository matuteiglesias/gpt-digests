# CSV Data Processing and Summary Generation

- message_id: `1e2cb596-cad7-4b6a-8b19-22a630260968`
- conversation_id: `690bc79c-026c-8332-9ace-87f99ae213f7`
- summary_day: `2025-11-05`
- category: Data
- format_type: script
- confidence: 4
- summary_tags: CSV | Data Processing | Pandas | Qualitative Analysis
- topics: Data | Automation

## Summary annotation

This script processes uploaded CSV files, performs qualitative scans, and generates summary tables including top tokens, bigrams, and sender statistics, while also displaying relevant dataframes to the user.

## Actions

Implement the CSV loading and analysis code in a data processing pipeline; explore monetization opportunities for CSV analysis tools.

## Notes

script

## Full text

### Chunk 1

- timestamp: `1762381318885`
- event_id: `690bc79c-026c-8332-9ace-87f99ae213f7:1e2cb596-cad7-4b6a-8b19-22a630260968`
- source_uri: `chatgpt:690bc79c-026c-8332-9ace-87f99ae213f7:1e2cb596-cad7-4b6a-8b19-22a630260968`

```text
# Loading uploaded CSV chunks, doing a lightweight qualitative scan, and producing summary tables.
import pandas as pd
from pathlib import Path
import re
from collections import Counter, defaultdict
import html
from caas_jupyter_tools import display_dataframe_to_user

# locate up to 10 part_*.csv files in /mnt/data
data_dir = Path("/mnt/data")
files = sorted(list(data_dir.glob("part_*.csv")))[:10]

if not files:
    raise FileNotFoundError("No part_*.csv files found in /mnt/data. Upload them first.")

# read and concatenate
dfs = []
for f in files:
    try:
        df = pd.read_csv(f)
        df['__source_file'] = f.name
        dfs.append(df)
    except Exception as e:
        print(f"Failed reading {f}: {e}")
if not dfs:
    raise RuntimeError("No CSVs could be read. Check format.")

df = pd.concat(dfs, ignore_index=True)

# Normalize columns - common names: sender_last4, sent_at_short, body or similar
possible_bodies = [c for c in df.columns if 'body' in c.lower()]
possible_senders = [c for c in df.columns if 'sender' in c.lower() or 'handle' in c.lower() or 'from' in c.lower()]
possible_time = [c for c in df.columns if 'sent' in c.lower() or 'time' in c.lower() or 'date' in c.lower()]

body_col = possible_bodies[0] if possible_bodies else (df.columns[0] if len(df.columns)>0 else None)
sender_col = possible_senders[0] if possible_senders else (df.columns[0] if len(df.columns)>0 else None)
time_col = possible_time[0] if possible_time else None

# Basic cleanup
df[body_col] = df[body_col].fillna('').astype(str)
if sender_col:
    df[sender_col] = df[sender_col].fillna('').astype(str)
if time_col:
    df[time_col] = df[time_col].fillna('').astype(str)

# Simple tokenization (Spanish + English stopwords)
stopwords = set("""de la que el en y a los del se las por un para con no una su al lo como más pero sus le ya o este sí porque esta entre cuando muy sin sobre también me hasta hay donde quien desde todo nos durante todos uno les ni contra otros ese eso había tanto ella fue poco ellos estar estas algunos muy""".split())
# add common WhatsApp noise tokens
stopwords.update({"https", "http", "www", "com", "img", "jpg", "png", "emoji"})

def tokenize(text):
    text = html.unescape(text)
    text = text.lower()
    # remove simple punctuation
    text = re.sub(r"[^\w\sáéíóúñü]", " ", text)
    tokens = [t for t in text.split() if t and t not in stopwords and not t.isdigit() and len(t)>1]
    return tokens

# gather counts
token_counter = Counter()
bigram_counter = Counter()
sender_counter = Counter()
samples_for_token = defaultdict(list)

for idx, row in df.iterrows():
    body = row[body_col]
    sender = row[sender_col] if sender_col else ""
    tokens = tokenize(body)
    token_counter.update(tokens)
    # bigrams
    bigrams = list(zip(tokens, tokens[1:]))
    bigram_counter.update([" ".join(bg) for bg in bigrams])
    sender_counter.update([sender])
    for t in set(tokens):
        if len(samples_for_token[t]) < 3:
            samples_for_token[t].append((idx, sender, body[:200]))

# Top tokens and bigrams
top_tokens = token_counter.most_common(30)
top_bigrams = bigram_counter.most_common(20)
top_senders = sender_counter.most_common(20)

# Activity over time (by date if time_col present)
if time_col:
    # try to coerce to datetime
    try:
        df['_dt'] = pd.to_datetime(df[time_col], errors='coerce', utc=True)
        df['_date'] = df['_dt'].dt.date
        activity_by_date = df.groupby('_date').size().reset_index(name='count').sort_values('_date')
    except Exception:
        activity_by_date = None
else:
    activity_by_date = None

# Prepare results as dataframes for display
df_top_tokens = pd.DataFrame(top_tokens, columns=['token','count'])
df_top_bigrams = pd.DataFrame(top_bigrams, columns=['bigram','count'])
df_top_senders = pd.DataFrame(top_senders, columns=['sender','count'])

# Show previews to user
display_dataframe_to_user("chunk_sample_preview", df.head(30))
display_dataframe_to_user("top_tokens", df_top_tokens.head(50))
display_dataframe_to_user("top_bigrams", df_top_bigrams.head(50))
display_dataframe_to_user("top_senders", df_top_senders.head(50))

if activity_by_date is not None:
    display_dataframe_to_user("activity_by_date", activity_by_date.head(200))

# Save short artifacts for the assistant's qualitative writeup
artifacts = {
    "n_files": len(files),
    "n_rows": len(df),
    "body_col": body_col,
    "sender_col": sender_col,
    "time_col": time_col,
    "top_tokens": top_tokens[:30],
    "top_bigrams": top_bigrams[:20],
    "top_senders": top_senders[:20],
    "samples_for_token": {k: v for k, v in list(samples_for_token.items())[:30]}
}

artifacts
```
