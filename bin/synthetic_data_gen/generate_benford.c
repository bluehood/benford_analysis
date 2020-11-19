#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h> 

		// Usage of the program

int usage(char* name_of_file)
{
	printf("\nUsage: Generate a Benford set of a given size and save to a file.\n\n%s %s %s %s %s\n\n", name_of_file, "<filename (str)>", "<size (int)>", "<lower magnitude (int)>", "<upper magnitude (int)>");
	printf("This program is designed to generate synethetic datra for single and multiple digit benford analysis on the first four digits!\n");
	printf("Lower and upper magnitude must be greater than or equal to four. The output is saved to filename and has size elements. Size must be a multiple of 1,000.\n");
	printf("This program is designed to run on Linux and has been tested on Ubuntu 18.04 LTS kernel version 5.4.0-51-generic x86_64.\n");
	return 0;
}


		// Random Number Generation 

// generate a random number between 0 and 1 
double rand_0_1()
{
    return rand() / ((double) RAND_MAX);
}

		// where the magic happens

void benford(FILE *outfile, int iterate, int lower_limit, int upper_limit)
{
	// generate inital random seed from /dev/random 
	int seed;
	FILE *f;
	f = fopen("/dev/random", "r");
  	fread(&seed, sizeof(seed), 1, f);
 	fclose(f);
	srand(seed);

	// Generate cumlative probability distribution of Benford distribution
	int cdf_size = 9001;
	double cdf[cdf_size];
	// set all entries to 0
	for (int n = 0; n < cdf_size; n++) 
	{
    	cdf[n] = 0.0;
    }
	// fill all elements, bar the first
	for (int n = 1; n < cdf_size; n++) 
	{
		double n_double = (double) n;
		cdf[n] = cdf[n - 1] + log10(1.0 + (1.0 / (n_double + 999.0)));
    }


	// get benford numbers in a list of 1000 and write them to a file each time
	int results[1000];
	for(int i = 0; i < iterate; i++)
	{
		// write the list of 1000 numbers to a file
		if ((i % 1000) == 0 && (i != 0))
		{
			for(int j = 0; j < 1000; j++)
			{
    			fprintf(outfile, "%d\n", results[j]);
			}
		}
		// gen random num between 0 and 1
		double rand_num = rand_0_1();

		// find a benford number
		for (int n = 1; n < 9001; n++) 
		{
			if ((cdf[n - 1]  <= rand_num) && (rand_num < cdf[n]))
			{	
				int magnitude = (rand() % (upper_limit + 1 - lower_limit)) + lower_limit;
				int final_num = (n + 999)*pow(10, magnitude);
				results[(i % 1000)] = final_num;
				break;
			}
		}

		// write the list of 1000 numbers to a file; outfile
		if(i == (iterate - 1))
		{
			for(int j = 0; j < 1000; j++)
			{
    			fprintf(outfile, "%d\n", results[j]);
			}
		}
	}
	
	fclose(outfile);
}

		// main function

int main(int argc, char *argv[])  
{  
	if(argc != 5)
	{
		fprintf(stderr, "Error: invalid arguements\n\n");
		usage(argv[0]);
		exit(EXIT_FAILURE);
	}
	else if(atoi(argv[2]) % 1000 != 0)
	{
		fprintf(stderr, "Error: argument specifying number of entries to generate must be a multiple of 1,000\n\n");
		usage(argv[0]);
		exit(EXIT_FAILURE);
	}
	else if((atoi(argv[3]) - 4) < 0 || (atoi(argv[4]) - 4) < 0)
	{
		fprintf(stderr, "Error: arguments specifying upper and lower limit of length must be >= 4\n\n");
		usage(argv[0]);
		exit(EXIT_FAILURE);
	}
	else
	{
		// take and store command line inputs
		char *filename = argv[1];
		int number_of_entries = atoi(argv[2]);	// number_of_entries must be a multiple of 1000 for this code to work
		int lower_limit = atoi(argv[3]) - 4;
		int upper_limit = atoi(argv[4]) - 4;

		// create file
		FILE *outfile;
		outfile = fopen(filename, "w");
		if (outfile == NULL)
		{
		fprintf(stderr, "Error: couldn't write to output file, potential invalid filename\n\n");
		usage(argv[0]);
		exit(EXIT_FAILURE);
		}

		// do the main computation
		benford(outfile, number_of_entries, lower_limit, upper_limit);

		// show we are done
		printf("Succesful output to %s\n\n", filename);
	}

	return 0;
}

// gcc -Wall -Wextra -pedantic  generate_benford.c -lm -o generate_benford

// ./generate_benford /home/odestorm/Documents/physics_project/analysis/data/synthetic/output.txt 100000 6 8

// python3 ../digit_test/benford.py /home/odestorm/Documents/physics_project/analysis/data/synthetic/output.txt 1 /home/odestorm/Documents/physics_project/analysis/data/synthetic/figures/output.png

// Console input format: filename, number of entries, lower limit magnitude, upper limit magnitude


