# Environment for Claude vibe-coding

## Workflow

### Communicate with me and Understand all my purpose

Before everything start, chat with me to understand all your jobs to do.

When you encounter choices/bugs, stop your work and chat with me.

### Start Coding

**!!! After Understanding All of what I told you !!!**

When starting to develop a feature, your workflow is:

- When starting to work:

```sh
# on branch main
git checkout main
git pull origin main
git checkout -b [Your Desired Branch]
```

- During your job, when finishing a milestone:

```sh
git add .
git commit -m "[Your Commits]"
```

- After your job

```sh
git add .
git commit -m "[Your Commits]"
git pull origin main --rebase
# merge all conflicts
git push origin [Your Desired Branch]
```

