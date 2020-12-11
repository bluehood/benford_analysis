# Git Commands

- git init
  - Starts git repo

-------------------------------

- git remote add origin ***the ssh from repository in github***
  - set destination in github for pushing and pulling

- git remote -v
  - shows destination of files in github

- git push origin ***branchname***
  - pushes branch up to github

- git pull origin ***branchname***
  - pulls branch in github to the local branch

-------------------------------

- git add ***filename***
  - adds file to commit

- git add -A
  - adds all files to commit that arent in it

- git commit -am ***"message"***
  - commit all changes to local repo and lets you set the message

- git status
  - shows the status of the commit in the current branch

- git log
  - shows a log of the commits and their descriptions

- git branch
  - shows all branches

- git show-branch
  - shows all branches

- git branch ***branchname***
  - make a branch of current branch
  
- git checkout ***branchname***
  - switches you to branch you chose

- git merge ***branchname***
  - merges branch into current branch

- git branch -d ***branchname***
  - Delete local branch 

- git push --delete origin ***branchname***
  - Delete remote branch

- git reset --soft HEAD~1
  - Delete most recent commit (on current branch)

- git push --all origin
  - Pushes all branches

- git log --color --graph --pretty=format:'%Cred%h%Creset -%C(yellow)%d%Creset %s %Cgreen(%cr)%C(bold blue) <%an> %Creset' --abbrev-commit
  - log visualisation

## Git Flow

- git flow init 
  - Initalises git flow in the git repository

- git flow feature start ***MYFEATURE***
  - start a feautre branch from the current branch

- git flow feature finish ***MYFEATURE*** 
  - merge the feature branch back into the branch the feature branched from and moves back into it, also deletes feature branch

-  git flow feature publish ***MYFEATURE***  
   -  push this feature branch to github