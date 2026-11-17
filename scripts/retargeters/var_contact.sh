#!/bin/bash
set -e
cd /home/user
python method/omniretarget.py --inputs "$1" --outputs "$2" --contact-weight 100
