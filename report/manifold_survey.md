# Survey: feature manifolds reported in language models (for the multi-prior "rainbow" experiments)

Compiled 2026-10-13 from the arXiv pages / TeX sources listed below. "Causal" = the paper intervenes along the
manifold and the model's output moves as the geometry predicts. Relation to our number helix: what is known (or
our own result) about whether the domain reuses the number subspace.

## 1. Cyclic (circle; group Z_n)

| domain | geometry | models | causal | relation to numbers | source |
|---|---|---|---|---|---|
| days of the week | circle (2-D) | GPT-2, Mistral-7B; used by Mistral-7B / Llama-3-8B for "k days after X" | yes | Llama-3.1-8B solves weekday arithmetic with its base-10 addition (with a +4 offset); OLMo-2: ablating the number subspace costs days only −5 to −7 (ours, F44) | Engels et al. ICLR 2025 (2405.14860); Feucht et al. 2026 (2605.01148) |
| months | circle | same | yes | Llama reuses base-10 addition; OLMo-2: number-subspace ablation −20 to −28 (F44) → **shared** | same |
| hours (24 h) | base-10 Fourier circles (T = 2, 5, 10), not a 24-cycle | Llama-3.1-8B | yes | **shared** (uses number tokens and number features) | Feucht et al. 2026 |
| musical chords / pitch | circle of fifths; dihedral (T/I) group structure on triads | Mistral Large 2, ChordBERT | – | unknown | Storan, Tolias, Miolane 2026 (2610.00647) |
| ordinal sequences (successor) | "mod-10 features" shared by numbers, number words, months, days, letters | GPT-2, Pythia, Llama-2 (31M–12B) | yes (numbers) | mod-10 features steer number tasks but **not** the days and letters tasks → letters likely separate | Gould et al. ICLR 2024 (2312.09230; TeX checked) |

## 2. One-dimensional ordered (line or curved line)

| domain | geometry | models | causal | source |
|---|---|---|---|---|
| years / dates | smooth 1-D curve ("rippled"), linearly decodable | Llama-2; text-embedding models | yes (release-year steering) | Gurnee & Tegmark ICLR 2024 (2310.02207); Karkada et al. 2026 (2602.15029); Modell 2026 (2605.18537) |
| number magnitude | log-compressed number line (Weber-like) | GPT-2 and others | – | 2502.16147; 2603.20642 |
| character count / line position | curved 1-D manifolds tiled by "place-cell" features; twisted by attention to compare | Claude 3.5 Haiku | yes | Gurnee et al. 2026 (2601.04480) |
| number comparison | curved geometry exists, but comparison uses a linear direction | Qwen2.5-7B-Instruct | yes | Hindupur et al. 2026 (2609.37680) |
| learned positions | helix (GPT-2 learned position embeddings) | GPT-2 | – | community analyses (LessWrong); not applicable to RoPE models such as OLMo-2 |

## 3. Two-dimensional and spherical

| domain | geometry | models | causal | source |
|---|---|---|---|---|
| world / US / NYC places | lat–long linearly decodable (plane) | Llama-2 | yes (neurons) | Gurnee & Tegmark ICLR 2024 |
| night sky (stars, constellations) | sphere (better: spherical harmonics); 65–85% variance, 12–21° median error | ~100B open models, Mistral Large 2 | – | Berdnikov & Liokumovich 2026 (2607.27092); Storan et al. 2026 |
| emotions | valence × arousal "circumplex" in the top PCs | Claude Sonnet 4.5; replicated on open models | yes (steering) | Anthropic 2026 (2604.07729); 2609.22208 |
| colours | alignment with perceptual (CIELAB) space; hue-circle topology in a translation model | text-only pretrained LMs; NLLB-200 | – | Abdou et al. CoNLL 2021 (2109.06129); 2603.02258 |

## 4. Simplices, belief states, in-context structures

| domain | geometry | models | source |
|---|---|---|---|
| categorical concepts | simplices; parent–child concepts in **orthogonal** subspaces (direct sums of simplices) | Gemma (WordNet, 957 concepts) | Park, Choe, Jiang, Veitch ICLR 2025 (2406.01506) |
| belief states over hidden states | probability simplex, fractal for Mess3 | small transformers on HMM data | Shai et al. NeurIPS 2024 (2405.15943) |
| posterior over Gaussian parameters | curved belief manifold, steerable | Llama-3.2 | Sarfati et al. 2026 (2602.02315) |
| graphs given in context (grid, ring, hexagon) | representations reorganise to the graph after enough context | open LLMs | Park et al. ICLR 2025 (2501.00070) |

## 5. Theory: where these shapes come from (useful for choosing targets)

- **Translation symmetry**: if the co-occurrence of two items depends only on their distance (months), the
  co-occurrence matrix is circulant, so the embedding is made of Fourier modes → circles; years become a rippled
  1-D curve; the geometry survives removing the direct co-occurrences (Karkada et al. 2602.15029).
- **Group invariance in general**: if a word family's statistics are invariant under a group G, the embeddings
  are matrix elements of G's irreducible representations: Z_12 → circles (months), dihedral → circle of fifths
  (chords), SO(3) → spherical harmonics (sky) (Storan et al. 2610.00647).
- **Features as metric spaces**: Modell, Rubin-Delanchy, Whiteley 2025 (2505.18235).
- Consequence for us: a target geometry need not be hand-designed. For a domain with symmetry group G, the target
  is G's irreps (all frequencies for Z_n). This matches our own results: multi-frequency targets help (F59) and
  more regular targets help more (F58).

## 6. Discovery tools (to find manifolds in our own checkpoints)

SAE feature clustering (Engels 2405.14860); Manifold Probe (Modell 2605.18537, code: alexandermodell/maniprobe);
SMIXAE (2605.09224); bilinear autoencoders (2510.16820); SAEs and manifold "tiling" / "dilution" (Bhalla et al.
2604.28119).

## 7. Prior work on imposing several manifold priors in training

Not found in a first search: REPA-style alignment exists for diffusion models (2410.06940) and diffusion LMs
(2605.06885), but it aligns to a teacher's representations, not to hand-specified concept geometries. "Concept-based
alignment" in 2610.01821 is a measurement across layers / models, not a training loss. A systematic novelty search
is still needed before writing.

## 8. Candidates for the rainbow, ranked by fit

Requirements: single-token items, a rule-based task with held-out pairs, a symmetry group (so the target follows
from §5), enough items for a stable Gram matrix (≥ 12), and known or measurable overlap with numbers.

| domain | items | task | target | expected overlap with numbers | fit |
|---|---|---|---|---|---|
| numbers 10..99 | 90 | addition | helix (periods 2/5/10/100 + linear) | – | anchor |
| letters A..Z | 26 | "F plus 3 letters is" / Caesar shift (Z_26) | line, or Z_26 irreps with wrap | low (mod-10 features do not steer letters) | high; already in concept_train |
| months | 12 | "March plus 5 months is" | Z_12 irreps | high (F44) | high; shared-manifold control |
| days | 7 | "Friday plus 3 days is" | Z_7 irreps | low–medium (F44) | medium (7 points) |
| Greek letters | 24 | offset within alpha..omega | line / Z_24 | unknown, probably low | medium; in number_route |
| pitch classes / chords | 12 | transpose by k semitones; circle of fifths | Z_12 or dihedral irreps | unknown; same group as months but a different concept | medium; tokenisation of C#, Eb to check |
| years 1900..1999 | 100 | "10 years after 1945" | line + period-10 | high (shares digits) | control only |
| colours | 6–12 | hue offsets / complements | hue circle | unknown | low (few items, naming not standard) |
| sky, places, emotions | many | knowledge-dependent | sphere / plane / circumplex | – | not suitable for early checkpoints or small pretraining |
