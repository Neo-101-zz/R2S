This directory contains the python code for reading WindSat V7 data.
Four files are needed: bytemaps.py, windsat_daily_v7.py, windsat_averaged_v7.py and example_usage.py

In order to test the programs, you need to 
1) download  the windsat_v7.0.1_verify.txt file located in the windsat/support_v07.0.1/verify/ directory
2) download the 4 files required to test:
For windsat:
Time              Directory                        File
daily passes      windsat/bmaps_v07.0.1/y2006/m04/       	wsat_20060409v7.0.1.gz
3-day mean        windsat/bmaps_v07.0.1/y2006/m04/      	wsat_20060409v7.0.1_d3d.gz
weekly mean       windsat/bmaps_v07.0.1/weeks/           	wsat_20060415v7.0.1.gz
monthly mean      windsat/bmaps_v07.0.1/y2006/m04/       	wsat_200604v7.0.1.gz

Each of these files contains WindSat data for:
the day April 9th, 2006 (daily)
the days April 7, 8 and 9th, 2006 (3-day mean)
April 9th (Sunday) to April 15th (Saturday), 2006 (weekly mean) 
or the month of April 2006 (monthly mean) 

3) place these files from step 1 and 2 in the same directory as the programs   
   
First run the daily and averaged routines to be sure they execute correctly.  You will get a 'verification failed' message if there is a problem.  If they work correctly, the message 'all tests completed successfully' will be displayed.

After confirming the routines work, use the example_usage.py routine as your base program and adapt to your needs.  This code shows you how to call the needed subroutines and diplays an example image.  Once you change the program, make sure to run it on the test files and check that the results match those listed in the windsat_v7.0.1_verify.txt file.

If you have any questions regarding these programs 
or the RSS binary data files, contact RSS support:
http://www.remss.com/support