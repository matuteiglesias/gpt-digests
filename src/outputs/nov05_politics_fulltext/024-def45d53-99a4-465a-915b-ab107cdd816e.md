# Message Processing Script for CSV Export

- message_id: `def45d53-99a4-465a-915b-ab107cdd816e`
- conversation_id: `690bc79c-026c-8332-9ace-87f99ae213f7`
- summary_day: `2025-11-05`
- category: Automation
- format_type: script
- confidence: 5
- summary_tags: Python | Pandas | Data Processing | CSV | Automation
- topics: Data | Automation

## Summary annotation

This script processes a DataFrame of messages, cleaning and formatting timestamps, handling NaN values, and chunking the output into CSV files. It includes a demo DataFrame for testing and demonstrates how to save processed data into specified output directories.

## Actions

Consider packaging the script as a reusable tool or library for data processing. Create documentation or a guide for users to implement similar processes.

## Notes

script

## Full text

### Chunk 1

- timestamp: `1762379859335`
- event_id: `690bc79c-026c-8332-9ace-87f99ae213f7:def45d53-99a4-465a-915b-ab107cdd816e`
- source_uri: `chatgpt:690bc79c-026c-8332-9ace-87f99ae213f7:def45d53-99a4-465a-915b-ab107cdd816e`

```text
# Demo and ready-to-run script: process messages, shorten timestamps, clean NaNs, keep last 4 digits of sender, and write chunked CSVs.
# If you already have `messags_pp` loaded in your environment, replace the demo creation section with your dataframe variable.
import os
import pandas as pd
from pathlib import Path

out_dir = Path('/mnt/data/wp_msg_dump')  # demo output dir in this environment
out_dir.mkdir(parents=True, exist_ok=True)

# --- DEMO DATAFRAME (remove this block if you already have `messags_pp`) ---
demo = {
    'sender_handle': ['+5491158893867', '+5491158893867', '+5491132640428', '+5491158893867', '+5491159046269'],
    'sent_at_utc': ['2021-09-20T19:00:28Z', '2024-07-10T16:22:33Z', '2024-07-10T16:22:41Z',
                    '2024-07-10T16:22:44.187000Z', '2024-07-10T16:24:07Z'],
    'body': ['💜 PARTICIPACIÓN 2024 💜', None, 'Lo agrego a Mati al grupo!', None,
             'Buenas! Nos juntamos mañana a las 13 30 en ext...']
}
messags_pp = pd.DataFrame(demo)
# --- END DEMO BLOCK ---

def process_and_chunk(df, out_dir='./wp_msg_dump', chunk_size=100, tz_local='America/Argentina/Buenos_Aires'):
    df = df.copy()
    # Parse timestamps (coerce invalids to NaT)
    df['sent_at_utc_parsed'] = pd.to_datetime(df['sent_at_utc'], errors='coerce', utc=True)
    # Try to convert to local timezone and produce a short human-readable string
    try:
        df['sent_at_short'] = df['sent_at_utc_parsed'].dt.tz_convert(tz_local).dt.strftime('%Y-%m-%d %H:%M')
    except Exception:
        # fallback: show UTC without seconds
        df['sent_at_short'] = df['sent_at_utc_parsed'].dt.strftime('%Y-%m-%d %H:%M UTC')
    # Replace NaN bodies with empty string and ensure plain text
    df['body'] = df['body'].fillna('').astype(str)
    # Take only last 4 characters of sender handle (safe even if shorter)
    df['sender_last4'] = df['sender_handle'].fillna('').astype(str).str[-4:]
    # Choose output columns
    out_df = df[['sender_last4', 'sent_at_short', 'body']].copy()
    # Ensure output directory exists
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # Chunk and save
    total = len(out_df)
    n_files = 0
    for i in range(0, total, chunk_size):
        part = out_df.iloc[i:i+chunk_size]
        n_files += 1
        filename = out_dir / f'part_{n_files:03d}.csv'
        part.to_csv(filename, index=False)
    return out_dir, n_files, out_df.head(10)

out_dir, n_files, preview = process_and_chunk(messags_pp, out_dir=out_dir, chunk_size=100)
print(f"Saved {n_files} file(s) into {out_dir}")
print("\nPreview of processed rows:")
print(preview.to_string(index=False))

# Show list of created files (up to 20)
files = sorted(list(out_dir.glob('part_*.csv')))
print("\nFiles created (showing up to 20):")
for f in files[:20]:
    print(f"- {f} (bytes: {f.stat().st_size})")

# If you'd like, the script created CSVs under the path(s) above. In your environment, change `out_dir` to './wp_msg_dump' if you want relative paths.
```
