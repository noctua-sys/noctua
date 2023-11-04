#!/bin/sh

mkdir -p logs

caffeinate -i &

./manage.py consistency --timeout=1000 --suffix=1s > logs/log1s &
./manage.py consistency --timeout=2000 --suffix=2s > logs/log2s &
./manage.py consistency --timeout=4000 --suffix=4s > logs/log4s &
./manage.py consistency --timeout=6000 --suffix=6s > logs/log6s &
./manage.py consistency --timeout=8000 --suffix=8s > logs/log8s &
./manage.py consistency --timeout=10000 --suffix=10s > logs/log10s &

wait
