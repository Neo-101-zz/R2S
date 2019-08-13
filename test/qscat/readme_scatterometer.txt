This directory contains access to Remote Sensing Systems' 
SeaWinds on Midori-II (SeaWinds) or SeaWinds on QuikSCAT (QuikSCAT)
Orbit Wind Vector Data (swath data) and 
Ocean Wind Bytemaps(0.25 degree gridded data).

!!This note has been updated on May 3,2011  !!!


*****Geophysical Model Function*****
The SeaWinds on QuikSCAT radar returns have been processed using the Ku-2011 geophysical model function.  This is referred to as version-4 data (_v4).  The data were reprocessed in late April 2011.

The SeaWinds on Midori-II remains the v3 version.  We will update this memo when the SeaWinds data are reporcessed.

Both QuikScat and SeaWinds winds agree extremely well with buoys typically showing a mean difference near 0 m/s with an rms below 1 m/s assuming no rain is present and the wind speeds are below 20 m/s.  


*****Data Files*******
There are two types of SeaWinds data available at this ftp site for both the
Midori-II and QuikSCAT platforms - Orbit files and Bytemap files.


****************
**Orbit Files***
****************
Orbit files are also known as wind vector files.  They are located in the seawinds_wind_vectors or qscat_wind_vectors directories.  Each file consists of one orbit of data.  The data are gridded into wind vector cells, 76 across the orbit and 1624 rows of cells along the orbit.

Reading subroutines are available in the scat_orbit_support directory
to assist in reading the orbit data files:
		get_scat_orbit_v04.pro		IDL subroutine
		get_scat_orbit_v04.f		Fortran subroutine
		get_scat_orbit_v04.m		Matlab subroutine  

Some users have had trouble on other systems with different compilers.  If you are having trouble reading the orbit files, check the specifics of how the file is opened and how the read is performed to make sure it matches the requirements of your compiler.


The orbit data files, once decoded, contain:
  ATIME is the 21 character time string, format:  YYYY-DDDTHH:MM:SS.sss
  PHI_TRACK is the direction of the subtrack velocity vector relative to north
  XLAT is the geodetic latitude
  XLON is the east geodetic longitude (0 - 360)
  
  ICLASS indicates the expected quality of the vector retrieval
  ICLASS=0 denotes no retrieval was done (either no observations or only one flavor of observation)
  ICLASS=1 denotes 2 flavors of observations in wind vector cell
  ICLASS=2 denotes 3 flavors of observations in wind vector cell
  ICLASS=3 denotes 4 flavors of observations in wind vector cell
  We suggest using just cases for which ICLASS.GE.2

  NUM_AMBIG is the number of ambiguites (0 to 4)
  ISELECT is the selected ambiguity (0 to 4)

  IRAIN_SCAT is the rain flag derived from the scatterometer measurements
  IRAIN_SCAT=1 indicates rain

  WINAL	is the 10 meter ocean wind speed for the various ambiguities
  PHIAL	is the wind direction for the various ambiguties (oceanographic convention)

  SOSAL	is the normalized rms after-the-fit residual of the observation minus model sigma-nought.  Large   SOSAL values indicate the observations did not fit the geophysical model function.
  We suggest discarding observations for which SOSAL.GT.1.9.

  WINDS	is the smoothed version of the selected wind, WINAL(ISELECT)
  PHIWS	is the smoothed version of the selected wind, PHIAL(ISELECT)

  WINGCM is the general circulation model wind speed used for nudging 
		(either NCEP or ECMWF)
  DIRGCM is the general circulation model wind direction used for 
		nudging (either NCEP or ECMWF)

  RAD_RAIN is the AMSR columar rain rate (rain rate times rain 
		 column height, km mm/hour)
    RAD_RAIN=-999     no AMSR rain avaliable
    RAD_RAIN=0        no rain
    RAD_RAIN=0.1      possible rain
    RAD_RAIN=0.2 through 25.4  definite rain and the given value is the columnar rain rate
   "no rain"  means no rain was detected within   +/- 50 km and +/- time given in MIN_DIFF.
   "possible rain"   means some rain was detected within +/- 50 km and +/- time given in MIN_DIFF.
   "definite rain"   means rain was detected within      +/- 25 km and +/- time given in MIN_DIFF.
  We suggest discarding observations for which RAD_RAIN.GT.0.15

  MIN_DIFF is the time difference in minutes between the scatterometer and the collocated radiometer.  
  A value of 255 means that no radiometer observation was collocated with the scatterometer.

  NUDGE_NCEP	
  NUDGE_ECWMF
  Each data file contains two INTEGER(1) variables at the end of the file. Both variables supply
  information on which GCM was used for the nudging field.  The first variable shows if NCEP data
  were used (0=used, 1=not used),and the second shows if ECMWF data were used (0=used,1=not used).
  Since GCM data are tri-linearly interpolated to the scatterometer data, it is possible that one
  of each map was used.  If both ECMWF and NCEP data were missing for the required orbit, the orbit
  is not processed since no nudging field would be available.

It is important to not use the data if the quality is suspect.  We recommend the 
following conditions be satisfied for the **very best** quality data:

  ICLASS(icel,iscan).ge. 2     (this would omit all outer swath data) 
  SOSAL(1,icel,iscan).le. 1.9  (yes, we mean the SOS of the first ambiguity)
  IRAIN_SCAT(icel,iscan).ne. 1
  MIN_DIFF(icel,iscan).le. 30 .and. RAD_RAIN(icel,iscan).le. 0.15


The SeaWinds on Midori-II dataset contains several gaps during the six months of 
data collection.  Official information on gaps and the cause of missing data 
can be found at http://podaac.jpl.nasa.gov/seawinds/seawinds_prob.html#gaps
The following are known gaps reported:
	revs	1724-1725
  		1895-1901
  		2241-2242
  		2396-2397
  		2774-2774
  		2877-2877
  		3173-3173
  		3382-3395
  		3471-3471
  		3476-3478
  		3647-3648
  		3786-3786
 		3871-3871
  		4453-4453

The SeaWinds on QuikSCAT dataset also contains gaps.  Official information
on gaps and the cause of missing data can be found at 
http://podaac.jpl.nasa.gov/quikscat/qscat_prob.html#gaps
The following are known gaps reported:
	revs	  2145   2171
		  2793   2810
		  3068   3070
		  5627   5636
		  6217   6218
		  7356   7385
		  9859   9907
		 10667  10694
		 12572  12599
		 14306  14329
		 15831  15836
		 16492  16504
		 17788  17807
		 20385  20389
		 22028  22029
		 23411  23423
		 26722  26729
		 36418  36426
		 36821  36855
		 38524  38527
		 40113  40115
                 40619  40621
                 44086  44096
                 46943  46962
                 49149  49190
                 49296  49302
                 53168  53196


The date and time information for each rev# is provided for each instrument in the 
SEAWINDS_INFO.txt and QSCAT_INFO.txt files
The columns in each file represent:
Rev #, # of good WVC rows, EQCROSSDATE, EQCROSSTIME, EQCROSSLONG, ORBITPERIOD

Due to the large number of files, we have subdivided them into directories of 
approximately 1000 files each.  01000to01999/		
				02000to02999/  etc.



*******************
***Bytemap Files***
*******************
Bytemap files are similar to those available from Remote Sensing Systems for SSMI and TMI.  They are located in the bmaps_v4 directory.
Each file consists of one day of data.  The data are gridded into 0.25 degree lat/lon cells, 1440 cells longitude by 720 cells latitude.

File reading subroutines are available in the scat_bmap_support directory
to assist in reading the bmap data files:
		get_scat_daily_v4.pro		IDL subroutines
		get_scat_averaged_v4.pro
		get_scat_daily_v4.f		Fortran subroutines
		get_scat_averaged_v4.f
		get_scat_daily_v4.m		Matlab subroutines
		get_scat_averaged_v4.f

Each of these subroutines have complementary main programs that show how to call
the subroutine and are already set up to write out the data shown in the
verification file.  Use the verification file to confirm that your adapted program
is still reading correctly after you have made any changes.

The bytemap files are (1440x720x4x2  or lon,lat,paramter,asc/dsc), 
and contain four parameters each stored as a single byte:
	1	Time (minutes of day GMT)   : TIME = IVAL * 6
	2 	Wind Speed	            : WSPD = IVAL*0.2
	3	Wind Direction		    : WDIR = IVAL*1.5
	4	Combination Rain Data	    : use bit extraction to access all data

	The combination rain flag contains a stand-alone scatterometer rain flag and available collocated radiometer rain data
	   bit pos 0       SCATEROMETER rainflag (0=no rain, 1=rain)
	   bit pos 1       collocated SSMI, TMI observations(for QuikSCAT data) 
	 	 	   or AMSR observations (for Midori-II data) within 60 min=1, else=0
           bit pos 2-7     0-63: radiometer rain where:
			   0= absolutely no rain 
			   1= rain in one of the adjacent cells
			   2= Radiometer RR = 0.5
			   3= Radiometer RR = 1.0
			   4= Radiometer RR = 1.5	 etc.
Each subroutine (IDL,Fortran,Matlab) performs the bit extraction and provides the 
user with the rain information.  It is important to remove rain contaminated data 
from your processing, especially at lower wind speeds typical of the tropics as 
this is where rain has the greatest effect. Rain causes SeaWinds to measure
higher wind speeds than are present in low wind situations.

If a day is missing from the data set it is because all orbits for that day are missing.
Official information on gaps and the cause of missing data can be found at the podaac web
site listed above.

Orbit numbers can be related to day and month bytemaps by looking at the SEAWINDS_INFO.txt
or QSCAT_INFO.txt files.  The date and time information for each rev# is provided in these files.
The columns represent:
Rev #, # of good WVC rows, EQCROSSDATE, EQCROSSTIME, EQCROSSLONG, ORBITPERIOD

The bytemap files are stored by year and month.  File names are qscat_YYYYMMDDv4.gz or YYYYMMDD.gz (for SeaWinds which is still v03)
Use any gzip compatible zip utility to uncompress the data prior to using the read subroutines provided.


*****Contact Information*****
If you have questions about RSS Scatterometer data, contact RSS support:
http://www.remss.com/support
