# This script will process downloaded csv from the compustat database giving postive and negative ea$
#!/bin/bash
ls ./positive || mkdir positive
ls ./negative || mkdir negative
for f in *.csv
do
    echo "Processing $f file"
    cat $f | cut -d ',' -f 8 | grep -v - > ./positive/positive_"$f"
    cat $f | cut -d ',' -f 8 | grep - | cut -c 2- > ./negative/negative_"$f"
done
