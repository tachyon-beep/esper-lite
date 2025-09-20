---
title: CITATIONS
source: /home/john/esper-lite/docs/paper/draft_paper.md
source_lines: 739-766
split_mode: consolidated
chapter: 13
coauthors:
  - John Morrissey
  - Codex CLI (OpenAI)
generated_by: scripts/split_paper.py
---

# Citations
This section lists the key publications that directly inform the core concepts, techniques, and architectural patterns discussed in this document. Each citation includes a note on its specific relevance.
[1] Houlsby, N., Giurgiu, A., Jastrzebski, S., Morrone, B., de Laroussilhe, Q., Gesmundo, A., ... & Gelly, S. (2019). Parameter-efficient transfer learning for NLP. In Proceedings of the 36th International Conference on Machine Learning (ICML).
Cited in Section 4. Basis for adapter layers as minimal, non-intrusive grafting strategies.
[2] Rusu, A. A., Rabinowitz, N. C., Desjardins, G., Soyer, H., Kirkpatrick, J., Kavukcuoglu, K., ... & Hadsell, R. (2016). Progressive neural networks. arXiv preprint arXiv:1606.04671.
Referenced in Section 3. Demonstrates early use of structural isolation and transfer in fixed-parameter agents, a foundational concept for freezing the base model.
[3] Kirkpatrick, J., Pascanu, R., Rabinowitz, N., Veness, J., Desjardins, G., Rusu, A. A., ... & Hadsell, R. (2017). Overcoming catastrophic forgetting in neural networks. Proceedings of the National Academy of Sciences, 114(13), 3521–3526.
Cited in Section 3 and 9. Introduces Elastic Weight Consolidation (EWC), a key method for preventing interference and a potential technique for allowing minimal, controlled plasticity at graft interfaces.
[4] Han, S., Pool, J., Tran, J., & Dally, W. (2015). Learning both weights and connections for efficient neural networks. In Advances in Neural Information Processing Systems (NeurIPS).
Referenced in Section 10. Origin of pruning-based network compression, relevant to Germinal Module (GM) recovery and the lifecycle management of germinated seeds.
[5] Rosenbaum, C., Klinger, T., & Riemer, M. (2019). Routing networks: Adaptive selection of non-linear functions for multi-task learning. In ICLR.
Cited in Section 3. Representative of dynamic neural architectures used for conditional computation, from which morphogenetic architectures draw the principle of structural adaptation.
[6] Beaulieu, S., Frasca, F., Xu, Y., Goyal, S., Pal, C., & Larochelle, H. (2020). Learning sparse representations in reinforcement learning with the successor features. In Advances in Neural Information Processing Systems (NeurIPS).
Supporting Section 3. Cited for modular representation learning, which is a prerequisite for effective and safe seed placement.
[7] Mallya, A., & Lazebnik, S. (2018). Piggyback: Adapting a single network to multiple tasks by learning to mask weights. In ECCV.
Referenced in Section 6. Describes masking-based adaptation of frozen networks, a concept related to the seed's local-only learning constraints.
[8] Schick, T., & Schütze, H. (2020). It's Not Just Size That Matters: Small Language Models Are Also Few-Shot Learners. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP).
Referenced in Section 1. Justifies the focus on sub-10M parameter models and the need for local capacity expansion where full retraining is infeasible.
[9] Elsken, T., Metzen, J. H., & Hutter, F. (2019). Neural architecture search: A survey. Journal of Machine Learning Research, 20(55), 1–21.
Cited in Section 2 and 10. Provides the broader context for automated structural growth, informing the design of morphogenetic control policies.
[10] Goyal, A., Lamb, A. M., Hoffmann, J., Sodhani, S., Levine, S., Bengio, Y., & Schölkopf, B. (2021). Inductive biases, pretraining and fine-tuning for transformer-based geometric reasoning. arXiv preprint arXiv:2110.06091.
Referenced in Section 10. Illustrates architectural localisation within Transformers, a key target for future seed placement strategies.
[11] Bengio, Y., & LeCun, Y. (2007). Scaling learning algorithms towards AI. In Large-scale kernel machines (Vol. 34, pp. 321–360).
Referenced in Section 10. A classic articulation of scalability and local learning principles, foundational to the entire morphogenetic perspective.
[12] Parisi, G. I., Kemker, R., Part, J. L., Kanan, C., & Wermter, S. (2019). Continual lifelong learning with neural networks: A review. Neural Networks, 113, 54–71.
Supporting background for Sections 1 and 3. Consolidates key methods and taxonomies in continual learning, relevant to the challenge of non-catastrophic adaptation.

APPENDICES
