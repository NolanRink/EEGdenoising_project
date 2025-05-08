I am using a Fabric Slice from the Kansas City Node with 1 A30 GPU, 16 processors, 128 GB of RAM, and 500 GB of disc space

Commands once in the Slice to run

ssh -F .\fabric_ssh_config -i .\newsliver ubuntu@2001:400:a100:3060:f816:3eff:fedb:ea85

sudo apt update
sudo apt install nvidia-driver-570

sudo reboot

nvidia-smi



