ssh -F .\fabric_ssh_config -i .\newsliver ubuntu@2001:400:a100:3060:f816:3eff:fedb:ea85

sudo apt update
sudo apt install nvidia-driver-570

sudo reboot

nvidia-smi
