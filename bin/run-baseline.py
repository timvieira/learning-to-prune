#!/usr/bin/env python
# http://wiki.clsp.jhu.edu/view/Grid

import subprocess
from path import path
from datetime import datetime
from hashlib import sha1
from argparse import ArgumentParser
from pandas import read_csv

p = ArgumentParser()
p.add_argument('--name', required=True)
p.add_argument('--dry', action='store_true')
p.add_argument('--skip-if-exists', action='store_true')
args = p.parse_args()


# results directory name is build on a hash of the command-line arguments

NAME = args.name #'baseline9'

for CLASSIFIER in ['LOGISTIC']:
    for TARGET in ['BODENSTAB_GOLD']:
        for C in [-13]:
            for GRAMMAR in ['big', 'medium']:

                init = read_csv('tmp/%s-%s.csv' % (NAME, GRAMMAR))
                tradeoffs = file('tmp/asym-%s.txt' % GRAMMAR).read().strip().split()

                for TRADEOFF in tradeoffs:

                    [thing] = init[init.args_initializer_penalty == float(TRADEOFF)].log
                    initweights = path(thing).dirname() / 'new_policy-001.npz'


                    cmd = ['python -u -m ldp.lols',
                           '--seed 0',
                           '--iterations 1',
                           '--tradeoff', TRADEOFF,
                           '--grammar', GRAMMAR,
                           '--maxlength 40',
                           '--minlength 3',
                           '--train 40000',
                           '--dev 10000',
                           '--accuracy evalb_corpus',
                           '--runtime pops',
                           '--minibatch 1000000000000',
                           '--classifier', CLASSIFIER,
                           '--initializer NONE',
                           '--initializer-penalty', TRADEOFF,
                           '--roll-out', TARGET,
                           '--init-weights', initweights,
                           '-C', C]

                    cmd = ' '.join(map(str, cmd))

                    date = datetime.now().strftime("%Y-%m-%d")

                    out = path('results/%s-%s-%s' % (date, NAME, sha1(cmd).hexdigest()))

                    cmd += ' --results %s/dump' % out

                    print cmd

                    if out.exists() and args.skip_if_exists:
                        print
                        print "[warning] skipping dump already exists."
                        print
                    else:

                        if args.dry:
                            qsub = ['echo', '\n\033[1;31m>>FAKEQSUB\033[0m']
                        else:
                            out.mkdir_p()
                            qsub = ['qsub']

                        sge_cmd = qsub + \
                            ['-cwd',
                             '-j', 'yes',
                             '-o', out / 'stdout.txt',      # stdout goes here.
                             '-b', 'yes',
                             '-N', NAME,                    # job name
                             '-l', "arch=*64*,mem_free=6G,ram_free=6G",  # flags for job requirements
                             '-q', "all.q@b*.clsp.jhu.edu", # run on b nodes they are much faster.
                             '/bin/bash',
                             'bin/sge-experiment',            # use wrapper script which has boilerplate for tracking jobid, starttime, etc.
                             out,
                             cmd]

                        #print ' '.join(map(str, sge_cmd))
                        print subprocess.check_output(map(str, sge_cmd))
