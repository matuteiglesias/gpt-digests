# Script for Timestamp Conversion and CSV Export

- message_id: `0ef645da-1fb8-4e70-80d5-d0f2328adfaa`
- conversation_id: `690bc79c-026c-8332-9ace-87f99ae213f7`
- summary_day: `2025-11-05`
- category: Programming
- format_type: script
- confidence: 5
- summary_tags: Python | Data Processing | CSV | Pandas
- topics: Scripting | Data Management

## Summary annotation

This message details a script that converts timestamps to a readable format, replaces NaN values, extracts the last four digits of sender handles, and saves messages in CSV files in chunks of 100. It includes a technical summary of the script's functionality and considerations for usage.

## Actions

Run the script on the user's CSV data or adjust the format as requested.

## Notes

script

## Full text

### Chunk 1

- timestamp: `1762379875245`
- event_id: `690bc79c-026c-8332-9ace-87f99ae213f7:0ef645da-1fb8-4e70-80d5-d0f2328adfaa`
- source_uri: `chatgpt:690bc79c-026c-8332-9ace-87f99ae213f7:0ef645da-1fb8-4e70-80d5-d0f2328adfaa`

```text
Listo — te armé y ejecuté un script de ejemplo que hace exactamente lo que pediste: convierte timestamps a una versión corta y legible, reemplaza NaN por `''`, toma los últimos 4 dígitos del número y guarda los mensajes en archivos CSV por bloques de 100. Ejecuté el ejemplo con una mini-muestra (igual a la que pegaste) y guardé `part_001.csv` en el entorno de la demo.

Qué hizo el script (resumen técnico rápido)
- `sent_at_utc` → parseo con `pd.to_datetime(..., utc=True)`.
- Conversión a zona horaria `America/Argentina/Buenos_Aires` y formateo `YYYY-MM-DD HH:MM` (si falla, queda en `YYYY-MM-DD HH:MM UTC`).
- `body`: `NaN` → `''` y forzado a `str`.
- `sender_handle`: toma `str[-4:]` (seguro si tiene menos de 4 caracteres).
- Agrupa en partes de 100 y escribe `part_001.csv, part_002.csv, ...` (nombre con padding `part_001.csv`).

Salida de la demo (preview):
sender_last4 | sent_at_short | body
-------------|---------------------|-----------------------------------------
3867 | 2021-09-20 16:00 | 💜 PARTICIPACIÓN 2024 💜
3867 | 2024-07-10 13:22 |
0428 | 2024-07-10 13:22 | Lo agrego a Mati al grupo!
3867 | 2024-07-10 13:22 |
6269 | 2024-07-10 13:24 | Buenas! Nos juntamos mañana a las 13 30 en ext...

Archivo generado en esta sesión de demo:
[Descargar part_001.csv](sandbox:/mnt/data/wp_msg_dump/part_001.csv)

Código completo (lista para pegar en tu entorno)
```python
import os
import pandas as pd
from pathlib import Path

def process_and_chunk(df, out_dir='./wp_msg_dump', chunk_size=100, tz_local='America/Argentina/Buenos_Aires'):
 df = df.copy()
 df['sent_at_utc_parsed'] = pd.to_datetime(df['sent_at_utc'], errors='coerce', utc=True)
 try:
 df['sent_at_short'] = df['sent_at_utc_parsed'].dt.tz_convert(tz_local).dt.strftime('%Y-%m-%d %H:%M')
 except Exception:
 df['sent_at_short'] = df['sent_at_utc_parsed'].dt.strftime('%Y-%m-%d %H:%M UTC')
 df['body'] = df['body'].fillna('').astype(str)
 df['sender_last4'] = df['sender_handle'].fillna('').astype(str).str[-4:]
 out_df = df[['sender_last4', 'sent_at_short', 'body']].copy()
 out_dir = Path(out_dir)
 out_dir.mkdir(parents=True, exist_ok=True)
 total = len(out_df)
 n_files = 0
 for i in range(0, total, chunk_size):
 part = out_df.iloc[i:i+chunk_size]
 n_files += 1
 filename = out_dir / f'part_{n_files:03d}.csv'
 part.to_csv(filename, index=False)
 return out_dir, n_files
```

Cómo usarlo con **tu** dataframe `messags_pp`
- Si en tu sesión ya existe `messags_pp`, simplemente llama:
```python
out_dir, n_files = process_and_chunk(messags_pp, out_dir='./wp_msg_dump', chunk_size=100)
print(f"Saved {n_files} files in {out_dir}")
```
- Si querés otra zona horaria o formato, cambiá `tz_local` o la cadena de `strftime`.

Consideraciones y edge-cases (lo que conviene revisar)
- Si `sent_at_utc` no es parseable (strings muy rotas), quedará `NaT` y `sent_at_short` puede ser `NaT` → conviene revisar y/o filtrar antes.
- Si el cuerpo contiene comas o saltos de línea, los CSV estarán bien (`to_csv` los escapará) pero para inspecciones rápidas puede convenir exportar a TSV o a parquet.
- Si querés conservar el `sender_handle` entero además del `last4`, incluilo en `out_df`.
- Último archivo puede tener menos de 100 filas — es normal.
- Si preferís nombres relativos (`./wp_msg_dump/...`) en lugar de ruta absoluta, pon `out_dir='./wp_msg_dump'` (el demo usó `/mnt/data/wp_msg_dump` para que puedas descargar el resultado desde aquí).

Si querés, lo próximo que puedo hacer ahora:
- ejecutar el mismo script sobre tu `messags_pp` si pegas aquí un path a un CSV (o subís un CSV) — lo procesaría y devolvería los CSVs.
- o ajustar el formato (por ejemplo, `DD/MM/YY HH:MM`, incluir `sender_handle` completo, o exportar a parquet en lugar de CSV).

¿Con qué prefieres que continúe? (si ya querés que lo ejecute sobre tus datos, pegá el path o subí el CSV y lo corrijo y guardo directamente).
```
