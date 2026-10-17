# Experiment scripts

The scripts behind the controlled studies in the audit report (§15.17 transfer study, §15.18 broad-pool RL)
and README §5.18. Run them from the repository root; intermediate files go to `$WORK`
(default `work/experiments`, ignored by git). Interpreters are taken from `$PY` (training/eval env),
`$VLLM_PY` (env with vLLM, for the local 27B teacher) and `$TORCHRUN`.

| script | what it does |
|---|---|
| `transfer_chain.sh` | GRPO / RFT / teacher-SFT on set A, scored on A, B600 and the real held-out set |
| `bsame_chain.sh` | GRPO on problems from B600's own papers (same-paper transfer) |
| `simple_chain.sh`, `simple_followup.sh` | one-step versions of A/B, GRPO on them, and the curriculum check on the originals |
| `pilot_chain2.sh`, `gen_notes.py`, `notes_sft.py` | the reading-versus-practice knowledge-augmentation pilot |
| `v1314_chain.sh` | broad-pool RL with stepping stones under the strict (v13) and corpus (v14) protocols |
| `num_eval_pot.py`, `real6_eval.py` | paired acc@5% on several sets / the 6-seed real held-out evaluation (deployment decoding) |
| `locality.py`, `taxonomy_file.py`, `difficulty.py`, `knowledge_split.py` | KL and log-prob shifts, failure taxonomy, gains by solution length, shared-constant split |
| `nn_sim.py`, `sibling_sim.py` | nearest-neighbour word-Jaccard between test questions and a training pool |
| `recover_chunks.py` | recover source-paper excerpts from the pretraining corpus by chunk id |
| `assemble_transfer.py` | collect the results into `docs/rl_transfer_diagnosis.json` (read by `src/plots/grpo.py`) |
| `rft_sample.py`, `headroom.py` | rejection-sampling data; pass@k / maj@k headroom |
