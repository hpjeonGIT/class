# Title: Git: Become an Expert in Git & GitHub in 4 Hours
- Instructor: VideoLab by Jad Khalili

## Section 1: Introduction

1. Introduction to Course

2. What is Git?
- Source control
  - It is usually saved in a series of snapshots and branches, which you can move back and forth between
  - Allows you to distribute your file changes over time, preventing against data loss/damage by creating backup snaphosts, and managing complex project structures

3. Git vs Github

4. Installing Git

## Section 2: The Terminal

5. Update about Terminal Section

6. Section Introduction 

7. Introduction to Terminal

8. Moving b/w Directories

9. Working with Files & Directories

## Section 3: Git Basics

10. Section Introduction

11. Git Cheatsheet
- https://github.com/joshnh/Git-Commands

### Getting & Creating Projects

| Command | Description |
| ------- | ----------- |
| `git init` | Initialize a local Git repository |
| `git clone ssh://git@github.com/[username]/[repository-name].git` | Create a local copy of a remote repository |

### Basic Snapshotting

| Command | Description |
| ------- | ----------- |
| `git status` | Check status |
| `git add [file-name.txt]` | Add a file to the staging area |
| `git add -A` | Add all new and changed files to the staging area |
| `git commit -m "[commit message]"` | Commit changes |
| `git rm -r [file-name.txt]` | Remove a file (or folder) |

### Branching & Merging

| Command | Description |
| ------- | ----------- |
| `git branch` | List branches (the asterisk denotes the current branch) |
| `git branch -a` | List all branches (local and remote) |
| `git branch [branch name]` | Create a new branch |
| `git branch -d [branch name]` | Delete a branch |
| `git push origin --delete [branch name]` | Delete a remote branch |
| `git checkout -b [branch name]` | Create a new branch and switch to it |
| `git checkout -b [branch name] origin/[branch name]` | Clone a remote branch and switch to it |
| `git branch -m [old branch name] [new branch name]` | Rename a local branch |
| `git checkout [branch name]` | Switch to a branch |
| `git checkout -` | Switch to the branch last checked out |
| `git checkout -- [file-name.txt]` | Discard changes to a file |
| `git merge [branch name]` | Merge a branch into the active branch |
| `git merge [source branch] [target branch]` | Merge a branch into a target branch |
| `git stash` | Stash changes in a dirty working directory |
| `git stash clear` | Remove all stashed entries |

### Sharing & Updating Projects

| Command | Description |
| ------- | ----------- |
| `git push origin [branch name]` | Push a branch to your remote repository |
| `git push -u origin [branch name]` | Push changes to remote repository (and remember the branch) |
| `git push` | Push changes to remote repository (remembered branch) |
| `git push origin --delete [branch name]` | Delete a remote branch |
| `git pull` | Update local repository to the newest commit |
| `git pull origin [branch name]` | Pull changes from remote repository |
| `git remote add origin ssh://git@github.com/[username]/[repository-name].git` | Add a remote repository |
| `git remote set-url origin ssh://git@github.com/[username]/[repository-name].git` | Set a repository's origin branch to SSH |

### Inspection & Comparison

| Command | Description |
| ------- | ----------- |
| `git log` | View changes |
| `git log --summary` | View changes (detailed) |
| `git log --oneline` | View changes (briefly) |
| `git diff [source branch] [target branch]` | Preview changes before merging |

12. The Git workflow
- Repositories
  - Called 'repos', store the full history and source control of a project
  - Locally or hosted on Github
  - Somewhere other than local location is called a 'remote repository'
- Commit: similar to taking a snapshot of the current state of the project, then storing it on a timeline

13. Creating a new Repository
- `git init` # initializes as a repo

14. Adding & Removing Files
- `git status` # info of stage
- `git add some_file`
- `git status` # shows that what changes have been made for next commit
- `git add .` # add all files in the current working folder
  - After deleting files manually, run `git add .` to update the status prior to commit
- `git rm --cached some_file` # removes the file from the stage, not physically deleting. The file will not be tracked anymore
  - `git rm -rf somefile` will remove the file from stage and the disk

15. Your First Commit
- Commit: makes a snapshot and put in the source track tree
- `git config --global user.email "you@example.com"`
- `git config --global user.name "your name"`
- `git commit -m "my commit message"`
- `git log` # shows all your commits
- `git push`

16. Git Checkout
- `git checkout hash_no_commit`: Go back to the commit stage
- `git checkout master`: come back to the latest commit
```bash
$ git init
$ touch  file01
$ git add file01
$ git commit -m "step01"
$ touch file2
$ git add .
$ git commit -m "step02"
$ touch file3
$ git add .
$ git commit -m "step03"
$ git log
commit d1f07fd163cfb4cdf006dbd449df683b09a00b56 (HEAD -> master)
Author: ...
Date:   Sat Dec 2 12:34:51 2023 -0500
    step03
commit 0f9f8d6720b604e9755596a75a6293d31eb49197
Author: ...
Date:   Sat Dec 2 12:34:42 2023 -0500
    step02
commit c5c3d3bb7ee9ac9594cf23df4d510a83b4e898ed
Author: ...
Date:   Sat Dec 2 12:34:25 2023 -0500
    step01
$ git checkout 0f9f8d6720b604e9755596a75a6293d31eb49197
$ git log
commit 0f9f8d6720b604e9755596a75a6293d31eb49197 (HEAD)
Author: hpjeonGIT <bjeon11@gmail.com>
Date:   Sat Dec 2 12:34:42 2023 -0500
    step02
commit c5c3d3bb7ee9ac9594cf23df4d510a83b4e898ed
Author: hpjeonGIT <bjeon11@gmail.com>
Date:   Sat Dec 2 12:34:25 2023 -0500
    step01
$ git checkout master
Previous HEAD position was 0f9f8d6 step02
Switched to branch 'master'
$ git log
commit d1f07fd163cfb4cdf006dbd449df683b09a00b56 (HEAD -> master)
Author: ...
Date:   Sat Dec 2 12:34:51 2023 -0500
    step03
commit 0f9f8d6720b604e9755596a75a6293d31eb49197
Author: ...
Date:   Sat Dec 2 12:34:42 2023 -0500
    step02
commit c5c3d3bb7ee9ac9594cf23df4d510a83b4e898ed
Author: ...
Date:   Sat Dec 2 12:34:25 2023 -0500
    step01
```

17. Git Revert & Reset
```bash
$ git revert 0f9f8d6720b604e9755596a75a6293d31eb49197 # will nano editor opens. ^x to save
$ git log
commit 035ce2cd13a9d414e3ae6a75e08ea7a0569b6400 (HEAD -> master)
Date:   Sat Dec 2 12:43:52 2023 -0500
    Revert "step02"
    This reverts commit 0f9f8d6720b604e9755596a75a6293d31eb49197.
commit d1f07fd163cfb4cdf006dbd449df683b09a00b56
Date:   Sat Dec 2 12:34:51 2023 -0500
    step03
commit 0f9f8d6720b604e9755596a75a6293d31eb49197
Date:   Sat Dec 2 12:34:42 2023 -0500
    step02
commit c5c3d3bb7ee9ac9594cf23df4d510a83b4e898ed
Date:   Sat Dec 2 12:34:25 2023 -0500
    step01
$ ls
file01  file3  #<--------- file2 is gone now
$ git revert 035ce2cd13a9d414e3ae6a75e08ea7a0569b6400 # revert of revert. ^x to save the nano editor
[master 098291d] Revert "Revert "step02""
 1 file changed, 0 insertions(+), 0 deletions(-)
 create mode 100644 file2
$ git log
commit 098291db1683bfbaa7cd55e65feeea9a37f0afa1 (HEAD -> master)
Date:   Sat Dec 2 12:44:28 2023 -0500
    Revert "Revert "step02""
    This reverts commit 035ce2cd13a9d414e3ae6a75e08ea7a0569b6400.
commit 035ce2cd13a9d414e3ae6a75e08ea7a0569b6400
Date:   Sat Dec 2 12:43:52 2023 -0500
    Revert "step02"
    This reverts commit 0f9f8d6720b604e9755596a75a6293d31eb49197.
commit d1f07fd163cfb4cdf006dbd449df683b09a00b56
Date:   Sat Dec 2 12:34:51 2023 -0500
    step03
commit 0f9f8d6720b604e9755596a75a6293d31eb49197
Date:   Sat Dec 2 12:34:42 2023 -0500
    step02
commit c5c3d3bb7ee9ac9594cf23df4d510a83b4e898ed
Date:   Sat Dec 2 12:34:25 2023 -0500
    step01
$ ls
file01  file2  file3  #<---------- now file2 is back
$ git log --oneline
098291d (HEAD -> master) Revert "Revert "step02""
035ce2c Revert "step02"
d1f07fd step03
0f9f8d6 step02
c5c3d3b step01
```
- Revert of revert will nullify the change

18. Types of Git Reset
- Soft: for minor mistakes - typo in the message
- Mixed: default - when some files are commited
- Hard: completely removes any change
```bash
$ git reset --soft d1f07fd #<--- step3
$ git log --oneline
d1f07fd (HEAD -> master) step03
0f9f8d6 step02
c5c3d3b step01
```

19. Creating a .gitignore
```bash
$ cat .gitignore
# images
*.webp
# sample log file
log.txt.txt
# for folders
tmp/*
$ ls
1.webp  file01  file3  log.txt.txt
$ git add .
$ git status
On branch master
Changes to be committed:
  (use "git restore --staged <file>..." to unstage)
	new file:   .gitignore
	new file:   file3
```
- Addressed files are not tracked

## Section 4: Git Branches

20. Section Introduction

21. What are Branches & why should you use them?
- Git branches are a way to create separate development paths without overriding or creating copies of your project
- Branches can be added, deleted, and merged, just like regular commits

22. Working with Branches

23. Editing Branches
- `git checkout -b dev` # make a new branch dev
- `git branch -a` # shows all branches
- `git checkout master`  # switches to master branch
- `git checkout dev`   # switches to dev branch
- `git branch -d dev` # removes dev branch

24. Merging Branches
- `git checkout dev`
- Do some works
- `git checkout master`
- `git merge dev`: merge dev branch into master

## Sectin 5: Github

25. Section Introduction

26. What is Github?

27. Creating a GitHub account

28. Creating our First Github Repository

29. Viewing Other Repositories

30. Download Github Repositories

## Section 6: Using Git Remotely

31. Section Introduction

32. Creating a new Remote Repository

33. The Push & Pull System

34. Pushing & Pulling to & from a Github Repository
- `git pull origin master`

35. Deleting Remote Branches

## Section 7: Git GUI w/ SourceTree

36. Section Introduction

37. What is SourceTree?

38. Installing SourceTree

39. Settting up a new Repository

40. Introduciton to the SourceTree Environment

41. Stage & Commit

42. Interaction in SourceTree

43. Create and Remove Branches

44. Merge Branches

45. Push/Pull Requets


## Section 8: Goodbye

## Section 9: Bonus Lectures

# Title: Git & GitHub Crash Course: Create a Repository From Scratch!
- Instructor: Kalob Taulien

## Section 1: Introduction

1. Introduction

2. Stream these videos in full HD!

3. Starting with Git and Github
- Cloning an existing repo to local machine
  - git clone https://github.com/XXX.git my_git_ex

4. Example Repo URL

5. A Quick Message

6. Adding Files and staging them
- git status: shows the status
- git add Readme.md: Staging. Not pushing yet
- git commit -m "my message here": committing
- git push origin master: Pushing or updating master repo
- git diff Readme.md: diffing

7. Committing your work and viewing differences
- git log: shows whave have been done

8. A super helpful git command!
- git log --topo-order --all --graph --date=local --pretty=format:'%C(green)%h%C(reset) %><(55,trunc)%s%C(red)%d%C(reset) %C(blue)[%an]%C(reset) %C(yellow)%ad%C(reset)%n'

9. Developer Support

10. Your Task

11. Where to Go next

12. Bonus material and where to go next

## Git: Become an Expert in Git & GitHub in 4 Hours
- Instructor: VideoLab by Jad Khalili

