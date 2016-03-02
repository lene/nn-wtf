#!/usr/bin/env bash

# Runs the test suite repeatedly to check for errors that only turn up randomly.

num_runs=1000

i=0
fails=0
while [ ${i} -lt ${num_runs} ] ; do
    echo Runs: ${i} Failures: ${fails}
    i=$[$i+1]
    python ./nn_wtf/run_tests.py || fails=$[fails+1]
done
