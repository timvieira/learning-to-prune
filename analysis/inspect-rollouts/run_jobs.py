"""Generate a list of commands to run on the grid.

Expects the output of bin/frontier --save as input. That file tells us which
parameters files to execute rollouts for.

"""
import cPickle
import subprocess
import pandas as pd
from path import path
from argparse import ArgumentParser

p = ArgumentParser()
p.add_argument('frontier_log', help='output of the frontier script.')
p.add_argument('--baseline', action='store_true',
               help='if enabled use init-weights (do baseline rollouts).')
args = p.parse_args()


# $ ./bin/frontier --target lols10 --accuracy evalb_avg --runtime pops \
#   --baseline-is-init --interpolation linear --filter 'df.args_grammar=="medium"' \
#   --early-stop-dev-cheat --save lols10.medium.cp.csv


df = pd.read_csv(args.frontier_log)


for _, x in df.iterrows():
    #print x.args_results
    results = path(x.args_results)
    job_args = cPickle.load(file(results / 'args.pkl'))
    policy = '%s/new_policy-%03d.npz' % (x.args_results, x.iteration)
    job_cmd = 'python -u analysis/inspect-rollouts/worker.py %s %s' % (policy, '--baseline' if args.baseline else '')

    out = path('evaluation') / 'asym_v_lols' / policy.replace('/', '_')

    if not out.exists():
        out.mkdir()

#    qsub = ['echo', 'qsub']
    qsub = ['qsub']

    NAME = 'inspect1'
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
               job_cmd]

    print subprocess.check_output(sge_cmd)
