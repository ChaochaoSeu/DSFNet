# Third-party notices

## BasicTS

The `basicts/` package is adapted from [BasicTS](https://github.com/GestaltCogTeam/BasicTS). The local framework identifies itself as version 0.5.1. Upstream distributes BasicTS under the Apache License, Version 2.0; a copy is included at [licenses/BasicTS-Apache-2.0.txt](licenses/BasicTS-Apache-2.0.txt).

This release reduces package exports to components used by DSFNet, removes disabled perturbation and alternative data-loading code, and edits comments and documentation. Existing training, splitting, scaling, and metric calculations are retained. Modified framework files carry a modification notice.

## Additional source references

Existing source references in the optimizer and scheduler modules are retained. They identify EasyTorch optimizer builders, Hugging Face scheduler code, nanoGPT optimizer logic, and modded-nanogpt optimizer logic. Runtime dependencies retain their own licenses.

## Scope

The bundled third-party license is an attribution for the adapted framework. It does not assign a license to the DSFNet-specific implementation or the processed datasets. Dataset-specific source and redistribution terms were not present in the supplied processed dataset descriptions, so this package does not invent such terms.
