# Jotish's branch – Circuitree

My work for the Circuitree ACM Research project: finding circuits that do the same job using graph machine learning.

| Folder | What's inside |
|---|---|
| `bench_parser/` | Reads a `.bench` file into a graph and counts the gates. Test file: `c17.bench`. |
| `c17_variants/` | First try at making circuit variants with ABC on c17, checked with `cec`. |
| `iscas85_variants/` | Notebook that makes 10 equivalence-checked ABC variants per ISCAS'85 circuit, plus the dataset zip (100 `.bench` files for 10 circuits, c17 not included, and `variants_summary.csv`). |
| `fgnn2_eval/` | FGNN2 retrieval model: evaluation, fine-tuning (Exp A), input positions (Exp B), input-count rule (Exp C), strict grader, top-20 + CEC tool, weights, and poster images. Final test results: 0.869 hard / 0.783 full Hit@1, 92% with a top-20 CEC check. Details in its README. |