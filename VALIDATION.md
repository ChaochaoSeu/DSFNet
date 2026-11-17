# Release validation

Checks performed on 2026-11-17 with Python 3.11.13, PyTorch 2.5.1, EasyTorch 1.3.3, and the versions listed in `requirements.txt`. Runtime smoke checks used the CPU.

## Data integrity

- All 30 processed dataset files passed byte-size and SHA-256 validation against the source files.
- All ten data tensors have the expected float32 size and contain only finite values.
- All ten adjacency files contain numeric, finite matrices with the expected node dimensions; their pickle metadata contains no site-name or record-identifier collections.
- Train, validation, and test readers loaded every dataset successfully. The first and last available windows in each split had the expected shapes.

## Model and framework behavior

- All 53 Python files compiled successfully without generating bytecode files.
- All five DSFNet configurations imported using only this repository and the installed dependencies.
- The executable syntax trees of the 24 copied framework modules and all five model architectures matched their source counterparts after ignoring docstrings. Curated package export files and command-line entry points were checked separately.
- Each model completed a real-data forward pass, runner preprocessing and postprocessing, finite MAE/MAPE/RMSE calculation, backward pass, and Adam update.
- With identical weights and input tensors, each cleaned model produced exactly the same output as its source model: maximum absolute difference was zero.
- An NJRC smoke run completed one training epoch using two real windows per split, saved a best checkpoint, reloaded it through the evaluation launcher, and saved metrics and predictions. All smoke-run outputs were stored outside the release folder.
- Both command-line entry points displayed their help successfully.

## Repository preparation

- All 24,865 original project files were checked against the initial snapshot. No files outside `DSFNET_GIT` were added, removed, or modified; original text-file hashes were unchanged.

- Text files and filenames were scanned for Chinese text, local usernames and absolute paths, email addresses, common access-token formats, private-key headers, and credential assignments. No matches were found.
- Raw records, standalone site-information tables, editor settings, checkpoint history, analysis outputs, ablation modules, and other baseline implementations are excluded.
- A temporary Git repository successfully staged the release with Git LFS. All 20 binary dataset files were stored as LFS pointers in the index; no ordinary staged blob exceeded 100 MiB. Checkpoint outputs were ignored.
- No repository history, remote, account identity, or credentials were added to the release folder.

These checks establish data integrity and basic runtime compatibility. Full dataset training, benchmark reproduction, and a remote GitHub push were not performed.
