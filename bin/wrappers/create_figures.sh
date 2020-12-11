# This script will create all figures for csv files in current directory. 
#!/bin/bash
# Create folders if they do not exist already
ls ./FR_first_digit || mkdir FR_first_digit
ls ./first_digit || mkdir first_digit
ls ./first_second_digit || mkdir first_second_digit
ls ./second_digit || mkdir second_digit
ls ./heatmaps || mkdir heatmaps 

# Loop through and create figures
for f in *.csv
do
	name=$(echo "$f" | cut -f 1 -d '.')
	echo "Creating Figures for $f file"
	python3 /home/odestorm/Documents/physics_project/analysis/benford_analysis/bin/digit_test/benford.py "$f" f1 ./FR_first_digit/FR_first_digit_"$name".png
	python3 /home/odestorm/Documents/physics_project/analysis/benford_analysis/bin/digit_test/benford.py "$f" 1 ./first_digit/first_digit_"$name".png
	python3 /home/odestorm/Documents/physics_project/analysis/benford_analysis/bin/digit_test/benford.py "$f" 12 ./first_second_digit/first_second_digit_"$name".png
	python3 /home/odestorm/Documents/physics_project/analysis/benford_analysis/bin/digit_test/benford.py "$f" 12hn ./heatmaps/first_second_heatmap_"$name".png
	python3 /home/odestorm/Documents/physics_project/analysis/benford_analysis/bin/digit_test/benford.py "$f" 2 ./second_digit/second_digit_"$name".png
	
done


