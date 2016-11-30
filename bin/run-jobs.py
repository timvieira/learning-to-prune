#!/usr/bin/env python

# http://wiki.clsp.jhu.edu/view/Grid


import subprocess
from path import path
from datetime import datetime
from hashlib import sha1
from argparse import ArgumentParser
from arsenal.terminal import green

p = ArgumentParser()
p.add_argument('name')
p.add_argument('--dry', action='store_true')
p.add_argument('--skip-if-exists', action='store_true')
args = p.parse_args()

NAME = args.name

for f in (path('jobs') / NAME).glob('jobs-*'):
    print green % '==> %s <==' % f
    cmds = file(f).readlines()
    for cmd in cmds:
        cmd = cmd.strip()

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
                qsub = ['echo', '\n\033[1;31m$ fake-qsub\033[0m']
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
