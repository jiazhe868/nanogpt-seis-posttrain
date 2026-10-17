"""Build the two pilot SFT mixtures: notes on the B chunks (K) or on the A chunks (control C), each with the
same replay sample of the notation2 SFT data so the QA format is kept."""
import json, random, sys
notes = [json.loads(l) for l in open(sys.argv[1])]
B = {json.loads(l)['chunk_id'] for l in open('data/rl/transfer_B600.jsonl')}
A = {json.loads(l)['chunk_id'] for l in open('data/rl/transfer_A.jsonl')}
rng = random.Random(0)
replay = [json.loads(l) for l in open('data/splits/train_pot_notation2.jsonl')]
replay = rng.sample(replay, 3000)
asks = ["Summarize the key quantitative relationships in the paper \"{t}\".", "Write study notes on the paper \"{t}\".",
        "Explain the main results and equations of \"{t}\".", "What does the paper \"{t}\" establish, quantitatively?"]
def rows(sel):
    out = [{'instruction': rng.choice(asks).format(t=n['title'] or 'this study'), 'response': n['note'], 'type': 'notes',
            'chunk_id': n['chunk_id']} for n in notes if n['chunk_id'] in sel]
    return out
kb, ka = rows(B), rows(A)
n = min(len(kb), len(ka)); kb, ka = rng.sample(kb, n), rng.sample(ka, n)
for name, r in (('K_notesB', kb), ('C_notesA', ka)):
    mix = r + replay; rng.shuffle(mix)
    open(f'data/splits/pilot_{name}.jsonl', 'w').writelines(json.dumps(x, ensure_ascii=False) + '\n' for x in mix)
    print(name, len(r), 'note rows +', len(replay), 'replay rows')
