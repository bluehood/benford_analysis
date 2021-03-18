# benford_analysis
Data analytics toolset for testing conformity of data with Benford's law, data acquisition and processing and synthetic data generation. This project was written as part of my Master's project at University. 

## Benford's Law
Benford's law states, for certain numerical sets, that the integer one appears in the leading significant digit about 30% of the time with a decrease in the expected proportion of higher integers; in particular, the integer nine only appears about 5% of the time. Benford's law also predicts the distribution of second, third and higher order digits. 

## Capabilities
The toolset has multiple capabilites:

* Synthetic Benford set generation including bruteforce and geometric series techniques. This allows for finer control over the degree of conformity with Benford's law. 
* Introduce arbitary deviations into Benford sets to observe changes in conformity. In particular, introduction of rounding behaviour and digit pronunciation (exaggeration) for first and second digits. 
* Traditional automated Benford tests, including the first, second and first-second digit tests. Each test includes plots of the data, either histograms or discrete heatmaps, and various test statistics assessing conformity. 
* Finite range Benford's law including a generalised finite range implementation using synthetic Benford sets. 
* Other general statistics and miscellaneous tools. 

Each program has its own usage when run without any commandline variables. Any C programs have the compliation instructions commented towards the bottom of the program. The project was developed on Ubuntu 18.04 but should work across other operating systems. 

## Specific Funcions
This section describes the functions and purpose of tools within the different folders in ./bin. 

* **adding_deviations**: introduce arbitary deviations into Benford sets and plot the resultant test statistics as a function of the strength of the deviations. Deviations include (for first or second digit tests):
  * Rounding Behaviour: introduce an excess of the lowest digit and a deficiency of the highest order digit. E.g. for the first digit test, introduce more one's and fewer nines into the distribution. 
  * Digit Pronunciation: produce an excess of a given digit within the distribution. 

* **data_mining**: software to download, extract and sanitise financial data from HTML formatted statements from the federal reserve (relating the assests of the largest banks in the USA https://www.federalreserve.gov/releases/lbr/) and annual reports filed with the Securities and Exchange commission (SEC https://sec.report/). 
* **digit_test**: perform Benford tests on a list of numerical data. The software sanitises the supplied data, applies Benford's law for a desired digit test, computes relevant test statistics assessing conformity and produces a plot of the results (histogram or discrete heatmap). 

The avaliable digit tests are:
    * 1    First Digit
  * f1   First Digit Finite Range
  * 2    Second Digit
  * f2   Second Digit Finite Range
  * 12   First-Second Digit
  * 12h  First-Second Digit with heatmap
  * 12hn Normalised Residual First-Second Digit with heatmap
  * 23h  Second-Third Digit with heatmap
  * 23hn Normalised Residual Second-Third Digit with heatmap

* **figures**: create various figures used during the project, mainly relating to histogtram and scatter plots. 
* **general_stats**: software performing general statistics calculations. 
* **misc**: miscellaneous software to test general hypotheses when they arise. For example, sum_test_example.py test that the sum of probabilites in Benford's law are suitably normalised (sum to one). 
* **presentation**: used to create figures or assist with my project presentation. 
* **synthetic_data_gen**: create synthetic Benford sets using a brute-force application of Benford's law or using geometric series. Compute test statistics for a set of Benford sets. 
