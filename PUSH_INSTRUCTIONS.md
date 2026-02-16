# Push to GitHub / GitLab

The repo is initialized locally with an initial commit. **You** need to create the remote repository and push (login and auth are required).

## 1. Create the remote repository

- **GitHub**: Go to [github.com/new](https://github.com/new). Name it (e.g. `RSC_Conv` or `lsi-ula-experiments`). Do **not** add a README, .gitignore, or license (you already have them locally).
- **GitLab**: New project → Create blank project. Leave “Initialize with a README” unchecked.

## 2. Add the remote and push

From your machine, in the project directory:

```bash
cd /Users/apple/Documents/RSC_Conv

# Replace with your actual repo URL (HTTPS or SSH):
# GitHub:  https://github.com/YOUR_USERNAME/RSC_Conv.git
# or SSH:   git@github.com:YOUR_USERNAME/RSC_Conv.git
git remote add origin https://github.com/YOUR_USERNAME/RSC_Conv.git

git push -u origin main
```

If your default branch is `master` instead of `main`:

```bash
git branch -M main   # already on main
git push -u origin main
```

## 3. Authentication

- **HTTPS**: Git will prompt for username and password. For GitHub, use a [Personal Access Token](https://github.com/settings/tokens) as the password.
- **SSH**: Ensure your SSH key is added to your GitHub/GitLab account and use the `git@github.com:...` URL.

After this, `git push` and `git pull` will work as usual.
