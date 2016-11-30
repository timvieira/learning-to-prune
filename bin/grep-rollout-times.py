#!/usr/bin/env python
"""
Grep for rollout times in logged stdout.
"""

import re, cPickle
import pylab as pl
import numpy as np
from path import path
from arsenal.terminal import green, yellow
from arsenal.humanreadable import htime

from argparse import ArgumentParser
p = ArgumentParser()
p.add_argument('--roll-out')
p.add_argument('--grammar')
p.add_argument('--classifier')
p.add_argument('--show-train', action='store_true')
p.add_argument('--show-stdout', action='store_true')
p.add_argument('--show-last-iter-stdout', action='store_true')
p.add_argument('--no-plot', action='store_true')
p.add_argument('files', nargs='+')

_args = p.parse_args()


PROGRESS = re.compile('.*\r(.*?)$', flags=re.MULTILINE)
def rm_progress(x):
    return PROGRESS.sub(r'\1', x)

train_times = []
data = []
for f in sorted(map(path, _args.files)):

    if not (f / 'dump/args.pkl').exists():
        continue

    with file(f / 'dump/args.pkl') as h:
        args = cPickle.load(h)

    if _args.grammar is not None and args.grammar != _args.grammar:
        continue
    if _args.roll_out is not None and args.roll_out != _args.roll_out:
        continue
    if _args.classifier is not None and args.classifier != _args.classifier:
        continue

    print green % '==> %s <==' % f
    with file(f / 'cmd.txt') as g:
        print yellow % g.read()

    tradeoff = args.tradeoff

    if _args.show_stdout:
        with file(f / 'cmd-stdout.txt') as g:
            for x in g:
                if x.startswith(' '):
                    continue
                if 'Training...' in x or 'using a fix' in x or 'Not a display' in x or 'grammar' in x or 'Grammar' in x:
                    continue
                if not x.strip():
                    continue
                print rm_progress(x.rstrip())   # remove progress bar intermediates

    if _args.show_last_iter_stdout:
        # TODO: show last iteration's stdout
        with file(f / 'cmd-stdout.txt') as g:
            out = g.read()

        iters = out.split('Iter')
        last_iter = iters[-1].strip().split('\n')

        iter_num = last_iter[0] if len(iters) > 1 else '1'

        print green % 'Iter %s' % iter_num
        for x in last_iter[1:]:
            if x.startswith(' '):
                continue
            if (#'Training...' in x or
                'using a fix' in x or 'Not a display' in x or 'grammar' in x or 'Grammar' in x):
                continue
            if not x.strip():
                continue
            print rm_progress(x.rstrip())   # remove progress bar intermediates
        print

    with file(f / 'cmd-stdout.txt') as g:
        z = []
        for x in g:
            if 'rollouts 100.0%' in x:
                y = x.split('\r')[-1]
                [x] = re.findall('rollouts .* (\d\d):(\d\d):(\d\d)', y)
                a,b,c = map(int, x)
                s = 60*60*a + 60*b + c
                z.append(s)

        z = np.array(z, dtype=float)

        print '[info] tradeoff: %g' % tradeoff
        print '[rollout]', ' '.join(map(htime, z[1:]))  # first rollout doesn't count.

        # some machines are just slower than others. The first iteration should
        # be the same across all machines. So, we'll use that to "correct" for
        # the difference in constant factors across different machines.
        #z /= z[0]

        data.append((tradeoff, z / 60**2))

    if _args.show_train:
        # ---------------------------
        # training times
        with file(f / 'cmd-stdout.txt') as g:
            xs = re.findall('train \((\d+h)?(\d+m)?(\d+s|\d+)?', g.read())
        z = []
        for x in xs:
            a,b,c = [int(w[:-1]) if w else 0 for w in x]
            s = 60*60*a + 60*b + c
            z.append(s)
        z = np.array(z, dtype=float)
        print '[train]', ' '.join(map(htime, z))
        train_times.append((tradeoff, z / 60))

    print

if train_times:
    pl.figure()
    pl.title('training classifier')
    for tradeoff, z in train_times:
        pl.plot(z, alpha=0.5)
    pl.ylabel('minutes')
    pl.xlabel('iteration')

pl.figure()
for tradeoff, z in data:
    pl.plot(z, alpha=0.5)
pl.title('Rollout')
pl.ylabel('hours')
pl.xlabel('iteration')

if not _args.no_plot:
    pl.show()
