#!/usr/bin/env bash

ack big results/*-$1-*/cmd.txt --l \
    | xargs dirname \
    | linepy 'print line+"/cmd-stdout.txt"' \
    | xargs ack 'rollouts.*?ETA'
