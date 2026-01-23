# Environment for Claude vibe-coding

## Workflow

### Communicate with me and Understand all my purpose

Before everything start, chat with me to understand all your jobs to do.

When you encounter choices/bugs, stop your work and chat with me.

### Start Planning

Write your plans in [plans directory](../plans/)

### Start Coding

**!!! After Understanding All of what I told you !!!**

There are submodules in our repo, so the standard development procedure is complex and  need to be clearified.

Notice that you shall never modify submodule and main repo in one task.

When you try to commit your changes, save the explanation of your task to [record directory](../record/), filename include date in the front.

#### Developing in submodules

Before you make contributions to submodules, please

```sh
cd [submodule]
git checkout -b [Your Desired Submodule Branch]
git add .
git commit -m "[Your Commits]"
```

After you made contributions to submodules, please

```sh
cd [submodule]
# make sure you are on the right branch
git checkout -b [Your Desired Submodule ranch]
git add .
git commit -m "[Your Commits]"
git push origin [Your Desired Submodule Branch]
```

Notice that the submodule's "main" branch is also protected, do not directly push to them, their main branches include:

- megatron_enhanced
- te_enhanced
- verl_enhanced

You should not change the main repo Megatron-LM-AutoTuner before your change to submodule merged. After whatever you changed to submodule, use the unit test I told you or write a unit test to test it on remote machine. which means that on the remote machine,

```sh
cd ~/projects/Megatron-LM-AutuTuner/[submodule]
git checkout -b [Your Desired Submodule Branch]
git pull origin [Your Desired Submodule Branch]
```

If failed, find why, if your implementation is wrong, return to host machine, check your implementation. And commit-push after that.

After all finished, stop your work, let me review and merge.

After I merged, you can modify in the main repo.

#### Developing main repo

Since our project is quite complex with multiple submodules, you should keep these in mind.

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

- After your job, check all contents, if there are contents like `.DS_Store` which shall not be included, exclude and delete them.

In main repo [Megatron-LM-AutoTuner](../../)

```sh
git add .   # Use . to include all submodules
git commit -m "[Your Commits]"
git pull origin main --rebase
# merge all conflicts
git push origin [Your Desired Branch]
```

And you can test them in other machine:

```sh
ssh [5090-1] "cd ~/projects/Megatron-LM-AutoTuner && git pull [Your Desired Branch] --recurse-submodules"
```

To test:

```sh
ssh [5090-1] "docker exec -it megatron_autotuner_new 'cd /workspace/Megatron-LM-Autotuner && [your command]'"
```
