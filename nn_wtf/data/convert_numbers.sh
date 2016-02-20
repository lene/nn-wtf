for i in {0..9}; do convert -geometry 28x28 -type grayscale -colorspace gray $i.png GRAY:$i.raw; done
