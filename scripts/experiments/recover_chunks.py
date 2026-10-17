"""Recover the source text of chunk ids (same chunking and id rule as src/sft/generate_sft_data.py)."""
import json, sys, hashlib
sys.path.insert(0, '.')
from src.sft.generate_sft_data import chunk_text
want = set()
for p in sys.argv[2:]:
    for l in open(p):
        r = json.loads(l); c = r.get('chunk_id') or r.get('seed_chunk')
        if c and c != 'None': want.add(c)
print('wanted chunk ids', len(want), flush=True)
dom_src = {"fulltext", "arxiv", "eartharxiv", "openalex", "substack", "journals", "preprints"}
kws = {"earthquake", "seismic", "seismology", "fault", "tectonic", "tsunami", "seismometer", "epicenter", "hypocenter",
       "subduction", "lithosphere", "attenuation", "ground motion", "shear wave", "p-wave", "s-wave", "magnitude"}
found = 0
with open(sys.argv[1], 'w') as g:
    for i, line in enumerate(open('../nanogpt_seis/data/processed/train.jsonl', encoding='utf-8')):
        line = line.strip()
        if not line: continue
        try: doc = json.loads(line)
        except Exception: continue
        src = doc.get('source', 'processed'); text = doc.get('text', doc.get('full_text', ''))
        if not text or len(text) < 500 or src in ('general', 'fineweb_edu', 'wiki_general'): continue
        if src not in dom_src and not any(k in text[:2000].lower() for k in kws): continue
        for c_idx, c in enumerate(chunk_text(text, chunk_size=8000)):
            cid = hashlib.md5(f"{doc.get('id', i)}_{c_idx}_{c[:50]}".encode()).hexdigest()
            if cid in want:
                g.write(json.dumps({'chunk_id': cid, 'doc_id': doc.get('id'), 'title': doc.get('title', ''), 'source': src, 'text': c}, ensure_ascii=False) + '\n')
                found += 1
print('found', found, 'of', len(want))
