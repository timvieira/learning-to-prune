from os import environ
DISPLAY = True
if not environ.get('DISPLAY'):  # pragma: no cover
    import matplotlib
    print 'Not a display environment'
    matplotlib.use('Agg')
    DISPLAY = False
