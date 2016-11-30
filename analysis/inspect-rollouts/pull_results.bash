#!/usr/bin/env bash

# job logs:
#rsync -a --progress \
#      "timv@login2.clsp.jhu.edu:/export/a11/timv/ldp/evaluation/asym_v_lols/." \
#      "evaluation/asym_v_lols/." \
#      --exclude '.nfs*' \
#      $@ \
#      2>/dev/null || die "error: rsync failed."

# job output:
rsync -a --progress "timv@login2.clsp.jhu.edu:/export/a11/timv/ldp/results/." \
      "results/." \
      --exclude 'baseline*' \
      --exclude 'searn*' \
      --exclude '*.npz' \
      --exclude '*.inspect_rollouts.csv' \
      --exclude cmd-stdout.txt
