These IDL, FORTRAN 90, Python, and C++ Matlab routines are available 
to aid in reading the RSS bytemap daily and time-averaged data files.  
The read routines have been tested and work correctly within our 
PC environment. We do not guarantee that they work perfectly in a different
environment with different compilers.  If portability is a problem,
we suggest using the Python code.

We provide data from the WindSat instrument on Coriolis platform.
We process the data using the version-7.0.1 (V7.0.1) RSS algorithm to
produce ocean products consisting of sea surface temperature, 
ocean surface winds (10 meters height), atmospheric water vapor, 
cloud liquid water, rain rate, all-weather winds and wind directions.

Details of the binary data file format is located at
http://www.remss.com/windsat/windsat_data_description.html#binary_data_files

Please make sure you have version v7.0.1 files with file dates after July 2013.


The FORTRAN subroutines are located in read_windsat_subroutines_v7.0.1.f
read_wsat_day returns a 1440x720x9x2 real array called wsat_data.
A description of the data within this array is at the top of the 
subroutine.  read_wsat_averaged returns a 1440x720x8 real array
called wsat_data.  The 3-day, weekly and monthly binary files
can all be read with this one routine.  Just supply the correct
filename with path.  These routines have been tested with 
Compaq Fortran 90.  Data files must be unzipped prior to using these routines.
Use a file unzipper of your choice.


The IDL read_windsat_day_v7.0.1.pro routine requires a full path filename
and returns nine 1440x720x2 real arrays.  The time-averaged
data files are read using read_wsat_averaged_v7.0.1.pro.  This routine
returns eight 1440x720 real arrays.  A description is provided 
within the routine.  These routines have been tested with IDL 8.1
Data files do not need to be unzipped when using the /compress keyword
in the read call.


Matlab subroutines, read_wsat_day_v7.0.1.m and read_wsat_averaged_v7.0.1.m, 
function similar to the IDL routines listed above and have been tested 
with Matlab 7.12.   Data files must be unzipped before using the matlab 
routines.


The Python code consists of windsat_daily_v7.py and windsat_averaged_v7.py.  
Both require the use of bytemaps.py.  The example_usage.py code provides a main
program that you can use to test the data or adapt to your needs.
Description of file contents is provided at the top of each routine.


C++ read routines consist of windsat_daily.h, windsat_daily.cpp and windsat_example.usage.cpp
as well as averaged files for each.  The windsat_example.usage.cpp provides
a main program that can be used to test the data or use as a starting
point to adapt to your needs.  A description of the file contents is provided
at the top of each routine.  dataset.h and dataset.cpp are required 
for programs to work.


Once you have further developed one of these skeleton programs
to suit your processing needs, you may wish to use the verify file 
to confirm that the data files are still being read correctly.
See the windsat_v7.0.1_verify.txt file located in the verify subdirectory.


If you have any questions regarding these programs 
or the RSS binary data files, contact RSS support:
http://www.remss.com/support





