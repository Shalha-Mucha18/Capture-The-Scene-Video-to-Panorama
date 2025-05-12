# Introduction to Git and GitHub

Git is a distributed version control system that helps developers track changes in their code and collaborate with others. GitHub is a platform that hosts Git repositories online, enabling collaboration, version control, and code sharing.

#### Example Code
**Initialize a new Git repository:**
```bash
git init
```
**Clone a repository from GitHub:**
```bash
git clone https://github.com/username/repository.git
```
### Why use Git?
Git is used to manage source code changes, track versions, collaborate with others, and maintain a history of the code. It enables teams to work on the same project without overwriting each other's changes and offers features like branching and merging.
### Difference between Git and GitHub
* Git is the version control system used to track code changes locally on your machine.
* GitHub is a cloud-based platform for hosting Git repositories and sharing code with others, providing additional features like pull requests, issues, and continuous integration.

![alt text](<../OneDrive/Desktop/sample/WhatsApp Image 2025-05-03 at 00.24.19_2cb0a808.jpg>)

**Git commands (local usage):**
```bash 
git add .
git commit -m "Your commit message"
```
**GitHub commands (push changes to GitHub):**
```bash
git push origin main
```

## Getting Started with Git

### Installing Git

To start using Git, install it on your machine. Git is available for Windows, macOS, and Linux.

**Installation:**
- **Windows/macOS:** Download from [git-scm.com](https://git-scm.com/downloads) and run the installer.
- **Linux (Debian/Ubuntu):**
  ```bash
  sudo apt update
  sudo apt install git
  ```
- **Linux (CentOS/Fedora):**
  ```bash
  sudo yum install git
  ```
- **Verify Installation:**
  ```bash
  git --version
  ```

### Configuring Git

Set up your user identity for commits.

```bash
git config --global user.name "Your Name"
git config --global user.email "your-email@example.com"
git config --list  # Verify configuration
```

### Basic Git Terminology

- **Repository (repo):** A folder where Git tracks changes.
- **Commit:** A snapshot of the repository at a point in time.
- **Branch:** A parallel version of the repository for isolated changes.
- **Main:** The default branch name for most repositories.

---

# Working with Local Repositories

### Initializing a New Repository

```bash
git init
```

### Cloning an Existing Repository

```bash
git clone https://github.com/username/repository.git
```

### Checking File Status

```bash
git status
```

### Staging Changes

```bash
git add file.txt       # Add specific file
git add .              # Add all changes
```

### Committing Changes

```bash
git commit -m "Your commit message"
```

### Viewing Commit History

```bash
git log          # Full commit history
git log --oneline  # Compact view
```

---

# Working with GitHub

### Creating a Remote Repository

```bash
git remote add origin https://github.com/username/repository.git
```

### Pushing and Pulling

```bash
git push origin main  # Push changes
git pull origin main  # Pull latest updates
```

---

# Collaboration on GitHub

### Forks vs Clones

- **Fork:** Creates a personal copy on GitHub.
- **Clone:** Downloads a GitHub repo to your local system.

```bash
git clone https://github.com/username/repository.git
```

### Making Pull Requests (PRs)

After committing to your fork or branch, submit a PR via the GitHub UI.

### Reviewing PRs

Project maintainers review, comment, and merge pull requests.

---

# Advanced Git Topics

### Reverting Changes

```bash
git reset --hard HEAD~1  # Undo last commit (dangerous)
git revert <commit>      # Safely revert a commit
```

### Rebasing

```bash
git checkout feature-branch
git rebase main  # Replay changes onto main
```

### Stashing

```bash
git stash        # Temporarily save work
git stash pop    # Restore latest stash
```

### Tags and Releases

```bash
git tag v1.0
git push origin v1.0
```

### .gitignore

```gitignore
# Common entries
node_modules/
.env
*.log
```

```bash
git rm -r --cached .
git add .gitignore
git commit -m "Apply .gitignore rules"
```
