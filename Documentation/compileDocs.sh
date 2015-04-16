markdown $1.md > $1
htmldoc --cont --headfootsize 8.0 --linkcolor blue --linkstyle plain --format pdf14 $1 > $1.pdf
rm $1