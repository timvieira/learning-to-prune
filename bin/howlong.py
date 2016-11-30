#!/usr/bin/env python
from path import path
from dateutil.parser import parse
from arsenal.terminal import green, magenta
from arsenal.humanreadable import htime
from sys import argv


done = []
running = []
for x in map(path, argv[1:]):
    try:
        start = parse((x / 'start').text())
    except (OSError, IOError):
        continue                  # didn't run on the grid.

    try:
        t = parse((x / 'finish').text()) - start
        done.append([t, x])

    except (OSError, IOError):
        running.append([start, x])

print
print magenta % 'Done'
print magenta % '======================================'
done.sort()
for t, x in done:
    print green % x, htime(t.seconds)

print
print magenta % 'Running'
print magenta % '======================================'
running.sort()
for t, x in running:
    print green % x, t, htime((t.now() - t.replace(tzinfo=None)).total_seconds())
