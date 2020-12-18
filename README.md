# class
- Collection of online classes

# git commands
- master -> main
- @local
  - git init
  - git add .
  - git commit -m "first upload"
  - git branch -M main
  - git remote add origin https://github.com/hpjeonGIT/class.git
  - git push origin main
  - git pull origin main

# Adding ssh key
- ssh-keygen -t ed25519 -C "__email_address__"
- eval "$(ssh-agent -s)"
- ssh-add ~/.ssh/id_ed25519
  - Add passphrase for higher security
- sudo apt-get install xclip
- xclip -selection clipboard < ~/.ssh/id_ed25519.pub
- Login to github -> settings -> add ssh-key
