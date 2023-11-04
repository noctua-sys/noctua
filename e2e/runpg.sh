#!/bin/bash

APP_DIR_BASE=../apps/PostGraudation
DB_FILE=soir-dev.db
CONFIG=pg.json
SECS=120
SC_FLAG=--sc
echo SC flag = $SC_FLAG
WRITE_RATIO=$1
N_SITES=3
echo Running ${N_SITES} sites!
EXP_NAME=SC-PG-w${WRITE_RATIO}-${SECS}s-$2

OUTDIR="${EXP_NAME}"
mkdir $OUTDIR || exit 1

REALOUTDIR=`realpath $OUTDIR`
echo $REALOUTDIR

PIDS=()

# Coordinator
python3 -m Coord.server --config=${CONFIG} ${SC_FLAG} > $OUTDIR/coord.log 2>&1 &
PIDS+=($!)
echo Coordinator started

# Sites
for site in $(seq $N_SITES)
do
    pushd ${APP_DIR_BASE}${site}
    python3 manage.py e2e -s ${site} > $REALOUTDIR/server${site}.log 2>&1 &
    PIDS+=($!)
    popd
done

echo Reinitialize databases
for site in $(seq $N_SITES); do
    rm $APP_DIR_BASE${site}/${DB_FILE}
    cp $APP_DIR_BASE/${DB_FILE} $APP_DIR_BASE${site}/${DB_FILE}
done

echo Sites started, waiting for bootstrapping...

sleep 10
echo Wait finished
echo Starting agents... Please wait for $SECS seconds

# Agent
AGENT_PIDS=()

for site in $(seq $N_SITES)
do
    ( python3 agent.py --config=$CONFIG --secs=$SECS --suffix=time --site=$site --write-ratio=$WRITE_RATIO ${SC_FLAG} > "$OUTDIR/agent${site}.log" 2>&1 ) &
    AGENT_PIDS+=($!)
done

for pid in ${AGENT_PIDS[@]}; do
    wait $pid
done

echo Agent finished

echo Stopping coordinator and sites...

for pid in ${PIDS[@]}; do
    kill $pid
done

# Merge results
touch $OUTDIR/merge.csv
for site in $(seq $N_SITES); do
    cat ${site}-time.csv >> $OUTDIR/merge.csv
    mv ${site}-time.csv $OUTDIR
done

