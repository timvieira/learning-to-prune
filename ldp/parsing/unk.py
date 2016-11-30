"""
Port of Berkeley parsers unknown word tools

Taken from the 2009-12-17 version of Slav Petrov's
edu.berkeley.nlp.PCFGLG.SophisticatedLexicon.java of the BerkeleyParser
(downloaded from http://code.google.com/p/berkeleyparser)

WARNINGS:
 - Has not been regression-tested against the original.
 - Only implements unknown level 5

"""

def signature(word, loc, lowercase_known):
    """
    This routine returns a String that is the "signature" of the class of a
    word. For, example, it might represent whether it is a number of ends in -s. The
    strings returned by convention match the pattern UNK-.* , which is just assumed
    to not match any real word. Behavior depends on the unknownLevel (-uwm flag)
    passed in to the class. The recognized numbers are 1-5: 5 is fairly
    English-specific; 4, 3, and 2 look for various word features (digits, dashes,
    etc.) which are only vaguely English-specific; 1 uses the last two characters
    combined with a simple classification by capitalization.

    @param word
                The word to make a signature for
    @param loc
                Its position in the sentence (mainly so sentence-initial capitalized
                words can be treated differently)
    @return
                A String that is its signature (equivalence class)
    """

    sb = ["UNK"]

    if len(word) == 0:
        return ''.join(sb)

    # Reformed Mar 2004 (cdm); hopefully much better now.
    # { -CAPS, -INITC ap, -LC lowercase, 0 } +
    # { -KNOWNLC, 0 } + [only for INITC]
    # { -NUM, 0 } +
    # { -DASH, 0 } +
    # { -last lowered char(s) if known discriminating suffix, 0}
    wlen = len(word)
    numCaps = 0
    hasDigit = False
    hasDash = False
    hasLower = False

    for i in xrange(wlen):
        ch = word[i]
        if ch.isdigit():
            hasDigit = True
        elif ch == '-':
            hasDash = True
        elif ch.isalpha():
            if ch.islower():
                hasLower = True
            elif ch.istitle():
                hasLower = True
                numCaps += 1
            else:
                numCaps += 1

    ch0 = word[0]
    lowered = word.lower()
    if ch0.isupper() or ch0.istitle():
        if loc == 0 and numCaps == 1:
            sb.append("-INITC")
            if lowercase_known:
                sb.append("-KNOWNLC")
        else:
            sb.append("-CAPS")

    elif not ch0.isalpha() and numCaps > 0:
        sb.append("-CAPS")

    elif hasLower:
        sb.append("-LC")

    if hasDigit:
        sb.append("-NUM")

    if hasDash:
        sb.append("-DASH")

    if lowered.endswith('s') and wlen >= 3:
        # here length 3, so you don't miss out on ones like 80s
        ch2 = lowered[wlen - 2]

        # not -ess suffixes or greek/latin -us, -is
        if ch2 != 's' and ch2 != 'i' and ch2 != 'u':
            sb.append("-s")

    elif len(word) >= 5 and not hasDash and not (hasDigit and numCaps > 0):
        # don't do for very short words
        # Implement common discriminating suffixes
        if lowered.endswith("ed"):
            sb.append("-ed")
        elif lowered.endswith("ing"):
            sb.append("-ing")
        elif lowered.endswith("ion"):
            sb.append("-ion")
        elif lowered.endswith("er"):
            sb.append("-er")
        elif lowered.endswith("est"):
            sb.append("-est")
        elif lowered.endswith("ly"):
            sb.append("-ly")
        elif lowered.endswith("ity"):
            sb.append("-ity")
        elif lowered.endswith("y"):
            sb.append("-y")
        elif lowered.endswith("al"):
            sb.append("-al")

    return ''.join(sb)
