# Third-party code and data

This file records the provenance and terms of third-party material used by the
simulator and its analyses. It does not grant a licence over the project's
original code, research outputs, or external datasets. The project author has not
selected a licence for the original code. The licences below retain their own
scope.

## SASRec adaptation

The SASRec model architecture is adapted from
[pmixer/SASRec.pytorch](https://github.com/pmixer/SASRec.pytorch), revision
`fde8a9c08febc2c898dc269b1d98ab762c50d2be`, under the Apache License 2.0.
The upstream software citation identifies Zan Huang's PyTorch implementation;
the original SASRec research is by Wang-Cheng Kang and Julian McAuley.

Affected files are `recommenders/sasrec.py` in `feedback_loop_experiment` and
`mitigation_experiment`, and `engine/recommenders/sasrec.py` in the two
`new_dataset_experiment` packages. The local adapters add simulator contracts,
explicit history and catalogue inputs, deterministic sampling, checkpoints,
and the configured mitigation loss where applicable. They are adaptations,
not unchanged upstream releases or published upstream model checkpoints.

- Exact retained licence: [LICENSES/SASRec-Apache-2.0.txt](LICENSES/SASRec-Apache-2.0.txt).
- Pinned source and modification record: [LICENSES/SASRec-PROVENANCE.md](LICENSES/SASRec-PROVENANCE.md).
- [Upstream model source](https://github.com/pmixer/SASRec.pytorch/blob/fde8a9c08febc2c898dc269b1d98ab762c50d2be/python/model.py).

The retained licence and provenance files are collected in the root `LICENSES/`
directory; references in source comments identify the original implementation.

## Agent4Rec MultVAE adaptation

`feedback_loop_experiment/recommenders/multvae.py` and the corresponding
`mitigation_experiment` module adapt the MultVAE network, loss, and
initialisation from the [Agent4Rec implementation](https://github.com/LehengTHU/Agent4Rec/blob/master/recommenders/models/MultVAE.py).
Agent4Rec distributes this code under the MIT License, with the upstream notice
`Copyright (c) 2023`. The notice does not name a copyright holder, and no name has
been added here. Simulator interfaces and training orchestration are local
adaptations.

- Exact retained licence: [LICENSES/Agent4Rec-MIT.txt](LICENSES/Agent4Rec-MIT.txt).
- [Upstream licence](https://github.com/LehengTHU/Agent4Rec/blob/master/LICENSE).

Agent4Rec is also a conceptual reference for the local agent's separation of
profile, planning, action, reflection, and memory. The local agent module is not
the upstream Agent4Rec runtime. Agent4Rec datasets, generated movie summaries,
and published model checkpoints are not covered by this source-code attribution.

## Lusifer profile prompts

The `persona_extraction/lusifer_adapter.py` modules preserve or adapt profile
instruction wording from [Danial Ebrat's Lusifer](https://github.com/danialebrat/Lusifer),
pinned revision `83a40a9c37ad8edc504321393964419f8505e0a1`. The Goodreads copies
adapt the instructions to books and restrict the supplied information to the
configured training histories. The adapters do not import the upstream Lusifer
runtime; the vendored upstream repository is omitted from this package.

The upstream README declares the project to be MIT-licensed. The recorded pinned
snapshot did not contain the linked `LICENSE` file. The original local working
copy therefore supplied a reconstructed MIT notice with the line
`Copyright (c) 2024 Danial Ebrat`. That line is a local provenance-based
reconstruction, not a verified upstream copyright notice.

Both local records are preserved without modification and clearly named:

- [LICENSES/Lusifer-PROVENANCE.md](LICENSES/Lusifer-PROVENANCE.md).
- [LICENSES/Lusifer-LOCAL-RECONSTRUCTION-MIT.txt](LICENSES/Lusifer-LOCAL-RECONSTRUCTION-MIT.txt).
- [Upstream MIT declaration](https://github.com/danialebrat/Lusifer#license).

The reconstructed file must not be represented as an upstream-authored licence
file. The missing authoritative notice remains a provenance qualification for
the copied prompt wording.

## Independently implemented methods

The local NCM modules record an independent implementation of architectures
described in *Neural Click Models for Recommender Systems*. They cite
[the paper's repository](https://github.com/arabel1a/Neural-Click-Models-for-Recommender-Systems)
for methodological provenance; that repository's notebook code and datasets are
not bundled or imported by these modules. The rank-based response rule and
mitigation methods likewise retain the scientific references stated in the
thesis and source. A methodological reference is distinguished here from a
copied source-code component.

## Dataset terms and full-run artifacts

Source datasets have terms separate from all source-code licences above.
Installing the code or receiving an experiment archive does not grant permission
to redistribute those datasets.

### MovieLens-1M

MovieLens-1M is provided by GroupLens Research at the University of Minnesota.
Its [official usage licence](https://files.grouplens.org/datasets/movielens/ml-1m-README.txt)
allows research use subject to conditions including acknowledgement, no implied
endorsement, no redistribution without separate permission, and permission for
commercial or revenue-bearing use. Obtain the dataset from the
[official dataset page](https://grouplens.org/datasets/movielens/1m/) and follow
its current terms. Cite Harper and Konstan (2015), *The MovieLens Datasets:
History and Context*, DOI `10.1145/2827872`.

### UCSD Goodreads Book Graph

The [official UCSD dataset page](https://cseweb.ucsd.edu/~jmcauley/datasets/goodreads.html)
states that the datasets were collected for academic use and requests no
redistribution or commercial use. This covers the supplied book metadata,
user-book interactions, and review collections. Obtain the required source
files separately and follow the provider's terms. The requested research
citations are Wan and McAuley (RecSys 2018), *Item Recommendation on Monotonic
Behavior Chains*, and Wan et al. (ACL 2019), *Fine-Grained Spoiler Detection from
Large-Scale Review Corpora*.

### VIAF authority metadata

The Goodreads gender analyses use information from VIAF (Virtual International
Authority File), hosted by OCLC. The
[VIAF data-source page](https://viaf.org/viaf/data) identifies the
[Open Data Commons Attribution License 1.0](https://opendatacommons.org/licenses/by/1-0/).

This research repository contains information from VIAF (Virtual International
Authority File), made available under the ODC Attribution License.

The annotations remain provisional matches to recorded authority metadata;
they are not assertions that the project verified an author's identity or
gender. The source snapshot, matching policy, and coverage diagnostics are
recorded by the preparation and analysis tools.

### Local full-run archives

Full logs and checkpoints are kept in the Git-ignored `artifacts/private/`
area for local review and restoration. They can contain initial user-item
histories, ratings, copied metadata, descriptions, and generated profile or
response text. A simulation checkpoint is a complete state bundle, not merely
a trained weight tensor. The MovieLens and Goodreads restrictions therefore
matter to source-bearing payloads inside those bundles as well as to standalone
dataset files.

Do not add the private bundles to Git or a public release without the permissions
needed for the source data they contain. Public manifests and restoration tools
do not change those underlying terms. The dataset pages do not specifically
classify every possible generated response or learned parameter file; this
notice does not declare all such derived material either restricted or freely
redistributable as a class.

## Installed dependencies

Python packages and optional provider SDKs are installed separately through the
project's dependency configuration. Their own licence notices continue to apply;
this package does not replace them with a blanket project licence.
