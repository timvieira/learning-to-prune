from ldp.prune.example import Setup, Reward

s = Setup(train=10, dev=0, maxlength=10, minlength=3)

for i, e in enumerate(s.train):

    print
    print 'Example', i
    print e.sentence
    for x in e.nodes:
        print x, e.features[x]
