#!/bin/bash
# Installing `lcml` on Ubuntu 16

# OS libraries
sudo apt update
sudo apt install python3-minimal python3-pip sqlite3 p7zip-full python3-tk

# mount elastic block store
sudo mkfs -t ext4 /dev/xvdb
sudo mkdir /data
sudo mount /dev/xvdb /data
sudo chown ubuntu /data

# git convenience
git config --global alias.co checkout
git config --global alias.br branch
git config --global alias.ci commit
git config --global alias.st status

# shell environment
vim ~/.bashrc
# add lines
export LCML=/home/ubuntu/light_curve_ml
export PYTHONPATH=/home/ubuntu/light_curve_ml:$PYTHONPATH
alias python=/usr/bin/python3
alias pip=/usr/bin/pip3
sudo mount /dev/xvdb /data

# install lcml
cd $HOME
git clone https://github.com/lsst-epo/light_curve_ml.git
cd $LCML
mkdir logs
pip install -e . --user

# ensure plotting will work
python
>>> import matplotlib
>>> matplotlib.matplotlib_fname()
vim [fname]
# switch backend property to: Agg
