This directory contains the python code for reading QuikSCAT data.
Four files are needed: bytemaps.py, quikscat_daily_v4.py, quikscat_averaged_v4.py and example_usage.py

In order to test the programs, you need to 
1) download the appropriate verify.txt file located in the scatterometer_bmap_support/quikscat_verify directory
2) download the 4 files required to test:
For quikscat:
   monthly file  	qscat_200001v4.gz
   daily file	qscat_20000111v4.gz
   3-day file	qscat_20000111v4_3day.gz
   weekly file	qscat_20000115v4.gz
3) place these files from step 1 and 2 in the same directory as the programs   
   
First run the daily and averaged routines to be sure they execute correctly.  You will get a 'verification failed' message if there is a problem.  If they work correctly, the message 'all tests completed successfully' will be displayed.

After confirming the routines work, use the example_usage.py routine as your base program and adapt to your needs.  This code shows you how to call the needed subroutines and diplays an example image.  Once you change the program, make sure to run it on the test files and check that the results match those listed in the quikscat_verify.txt file.

If you have any questions regarding these programs 
or the RSS binary data files, contact RSS support:
http://www.remss.com/support