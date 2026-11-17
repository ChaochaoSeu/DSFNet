# Uploading to GitHub

All commands below must run inside this `DSFNET_GIT` folder. Upload this folder as the repository root.

## 1. Verify the files

```bash
python scripts/verify_datasets.py
```

The processed data occupy about 2.06 GiB. The largest data file is about 744 MiB. GitHub blocks ordinary Git files larger than 100 MiB, so the included `.gitattributes` tracks dataset binaries with Git LFS. Use Git with LFS rather than the browser's file-upload interface.

## 2. Create a local repository

Install [Git LFS](https://git-lfs.com/) if `git lfs version` is unavailable, then run:

```bash
git init -b main
git lfs install --local
git add .gitattributes .gitignore
git add .
git lfs ls-files
git diff --cached --stat
```

`git lfs ls-files` should list 20 files: ten `data.dat` files and ten `adj_mx.pkl` files. Keep `.gitattributes` in the first commit so binaries are handled by LFS from the beginning.

## 3. Use a public-safe commit identity

Git commits include an author name and email. Set a repository-local public alias and your GitHub-provided noreply email from **Settings > Emails** before committing:

```bash
git config --local user.name "PUBLIC_ALIAS"
git config --local user.email "YOUR_GITHUB_NOREPLY_EMAIL"
git commit -m "Initial DSFNet release"
```

Replace both placeholders. These commands do not change your global Git identity.

## 4. Push to an empty repository

Create an empty GitHub repository without an initial README, license, or `.gitignore`. Replace the remote URL below with that repository's URL:

```bash
git remote add origin https://github.com/ACCOUNT/REPOSITORY.git
git push -u origin main
```

Git LFS uploads the data objects as part of the push. The repository owner's LFS storage and bandwidth quotas apply. This package contains no remote URL, Git history, credentials, or configured commit identity.

## References

- [GitHub file size limits](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github)
- [Git Large File Storage](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-git-large-file-storage)
- [Commit email privacy](https://docs.github.com/en/account-and-profile/how-tos/email-preferences/setting-your-commit-email-address)
