#!/usr/bin/env python
import re, os
from arsenal.fsutils import find
from arsenal.terminal import green

OPTS = {
    'language': 'c++',
    'cdivision': 'True',
    'boundscheck': 'False',
    'libraries': "['stdc++']",
    'wraparound': 'False',
    'infertypes': 'True',
    'nonecheck': 'False',
    'initializedcheck': 'False',
#    'NPY_NO_DEPRECATED_API': 'NPY_1_7_API_VERSION',   # not the right check.
}

for x in sorted(list(find('.', glob='*.pyx')) + list(find('.', glob='*.pxd'))):
    print green % x
    with file(x) as f:
        header = [y.strip() for y in f.readlines()[:30] if y.startswith('#')]
        opts = {y:z for l in header for x,y,z in re.findall('#(cython|distutils):\s*(.*?)\s*=\s*(.*?)\s*$', l)}

#        defs = {y:z for l in header for x,y,z in re.findall('#(DEFINE)\s+(\S+)\s+(.*?)\s*$', l)}
#        opts.update(defs)

        for k in OPTS:
            exp = OPTS[k]
            got = opts.get(k)
            if exp != got:
                print '%s expected: %s got: %s' % (k, exp, got)
#                os.system('emacs -nw "%s" 2>/dev/null' % (f.name))

        for k in opts:
            if k not in OPTS:
                print 'extra option: %s = %s' % (k, opts[k])
