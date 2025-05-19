cd ~
git clone https://github.com/Homebrew/brew homebrew
mkdir ~/usr/local
# installed packaged directory
echo "export HOMEBREW_PREFIX=~/usr/local" >> ~/.zshrc
echo "export PATH=$PATH:~/homebrew/bin:HOMEBREW_PREFIX/bin" >> ~/.zshrc
