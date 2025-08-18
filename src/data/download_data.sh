#!/bin/bash

url="user@dsipshare.fbkeduroam.it:/media/user/wd2/datasets/climate/seasonal/"
rsync -a -e ssh $url .