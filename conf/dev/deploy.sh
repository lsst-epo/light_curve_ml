sudo apt update
sudo apt install python3-minimal python3-pip sqlite3 p7zip-full python3-tk
pip install --upgrade pip

>>> import matplotlib
>>> matplotlib.matplotlib_fname()
vim [fname]
# switch backend to: Agg

git config --global alias.co checkout
git config --global alias.br branch
git config --global alias.ci commit
git config --global alias.st status

cd $HOME
git clone https://github.com/carpyncho/feets.git
git clone https://github.com/lsst-epo/light_curve_ml.git

sudo mkfs -t ext4 /dev/xvdb      (‘xvdb’ maps to ‘sdb’ in the “Device” value in the screenshot above; change if needed)
sudo mkdir /data
sudo mount /dev/xvdb /data
sudo chown ubuntu /data

# copy dirs recursively into EBS mount
scp -rp -i ~/.ssh/rjm_lsst_2018.pem ogle3/ ubuntu@ec2-34-215-180-36.us-west-2.compute.amazonaws.com:/data

# in ~/.bashrc
vim ~/.bashrc
export LCML=/home/ubuntu/light_curve_ml
export PYTHONPATH=/home/ubuntu/light_curve_ml:$PYTHONPATH

alias python=/usr/bin/python3

cd $LCML
mkdir logs
pip install -r requirements.txt

cd $HOME/feets
pip install -e .
