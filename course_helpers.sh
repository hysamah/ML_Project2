#!/bin/bash     
declare DATA_PATH="twitter-datasets"
cat $DATA_PATH/train_pos_clean.txt $DATA_PATH/train_neg_clean.txt | sed "s/ /\n/g" | grep -v "^\s*$" | sort | uniq -c > vocab_full.txt
cat vocab_full.txt | sed "s/^\s\+//g" | sort -rn | grep -v "^[1234]\s" | cut -d' ' -f2 > vocab_cut.txt
