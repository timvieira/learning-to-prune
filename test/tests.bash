#!/usr/bin/env bash

#
# Script invokes several commands. Used by `test` script which compares against
# expected output from an older version of codebase.
#
set -e

# disable plotting
DISPLAY=

function green { echo -e "\e[32m$@\e[0m"; }

function run_test {
    echo ===========================================================
    green $1 1>&2
    echo $1
    time $1
    echo 1>&2
}

run_test "python test/test-feature-stability.py"
run_test "python -m ldp.parsing.evalb"
run_test "python -m ldp.math.test.tests"
run_test "python -m ldp.math.test.sparse"
run_test "python -m ldp.parsing.util"
run_test "python -m ldp.cp.test.tiny"
run_test "python -m ldp.dp.test.correctness --examples 20 --maxlength 15 --seed 0 --delta 0.5 --grammar medium"
run_test "python -m ldp.parsing.ptb"
run_test "python -m ldp.cp.test.correctness --seed 0 --maxlength 15 --examples 20 --aggressive 0.5"
run_test "python -m ldp.cp.test.correctness --seed 0 --maxlength 15 --examples 20 --aggressive 0.5 --boolean"

# run with Changeprop rollouts.
run_test "python -m ldp.lols
--grammar medium
--train 100 --dev 0 --maxlength 10
--iterations 10 --minibatch 10 --tradeoff 0.0005
--classifier PERCEPTRON -C -12
--accuracy evalb_corpus --runtime pops
--initializer BODENSTAB_GOLD
--roll-out CP
--seed 0"

# run with Changeprop rollouts.
run_test "python -m ldp.lols
--grammar medium
--train 100 --dev 0 --maxlength 10
--iterations 10 --minibatch 10 --tradeoff 0.0005
--classifier PERCEPTRON -C -12
--accuracy evalb_avg --runtime pops
--initializer BODENSTAB_GOLD
--roll-out CP
--seed 0 "

# run with DP-alg rollouts.
run_test "python -m ldp.lols
--grammar medium
--train 10 --dev 0 --maxlength 10
--dev 10
--iterations 10
--minibatch 10
--tradeoff 0.1
--classifier ADAGRAD -C -12
--accuracy expected_recall_corpus
--runtime mask
--initializer BODENSTAB_GOLD
--roll-out DP
--learning-rate 0.1
--seed 0"

# run with DP-alg rollouts.
run_test "python -m ldp.lols
--grammar medium
--train 10 --dev 0 --maxlength 10
--dev 10
--iterations 10
--minibatch 10
--tradeoff 0.1
--classifier ADAGRAD -C -12
--accuracy expected_recall_avg
--runtime mask
--initializer BODENSTAB_GOLD
--roll-out DP
--learning-rate 0.1
--seed 0"
