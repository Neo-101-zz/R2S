This directory contains access to Remote Sensing Systems' 
ASCAT Orbit Wind Vector Data (swath data) and 
Ocean Wind Bytemaps(0.25 degree gridded data)
for the ASCAT instrument on Metop-A

*****Geophysical Model Function*****
The ASCAT on MetOp-A recalibrated radar returns have been processed using the RSS C-2015 geophysical model function.  
Available as of April 2016, these data are referred to as version-02.1 data (_v02.1).  

The ASCAT data agree extremely well with buoys and QuikSCAT data typically
showing a mean difference near 0 m/s with an rms below 1 m/s when no rain is present.


*****Data Files*******
There are two types of ASCAT data available at this ftp site  - Bytemap files and Orbit files.

*******************
***Bytemap Files***
*******************
Bytemap files are similar to those available from Remote Sensing Systems for microwave radiometers.  
They are located in the bmaps_v02.1 directory.
Each file consists of one day of data.  The data are gridded into 0.25 degree lat/lon cells,
1440 cells longitude by 720 cells latitude.

File reading subroutines are available in the ascat_bmap_support directory
to assist in reading the bmap data files:
		get_ascat_daily.pro		IDL subroutines
		get_ascat_averaged.pro
		get_ascat_daily.f		Fortran subroutines
		get_ascat_averaged.f
		get_ascat_daily.m		Matlab subroutines
		get_ascat_averaged.f
		There are also C++ and Python read routines for the bytemap data

Each of these subroutines have complementary main programs that show how to call
the subroutine and are already set up to write out the data shown in the
verification file.  Use the verification file to confirm that your adapted program
is still reading correctly after you have made any changes.

The bytemap files are (1440x720x5x2  or lon,lat,paramter,asc/dsc), 
and contain four parameters each stored as a single byte:
	1	Time (minutes of day GMT)   : TIME = IVAL * 6
	2 	Wind Speed		    : WSPD = IVAL*0.2
	3	Wind Direction		    : WDIR = IVAL*1.5
	4	Combination Rain Data	    : use bit extraction to access all data
	5	SOSmap			    : SOSMAP = IVAL * 0.02, non-dimensional

	The combination rain flag contains a stand-alone scatterometer rain flag and available
	collocated radiometer rain data
	            bit pos 0       SCATEROMETER rainflag (0=no rain, 1=rain)
	            bit pos 1       collocated SSMI, SSMIS observations within 180 min =1, else=0
	            bit pos 2-7     0-63: radiometer rain where:
				0= absolutely no rain 
				1= rain in one of the adjacent cells
				2= Radiometer RR = 0.2
				3= Radiometer RR = 0.4
				4= Radiometer RR = 0.6	 etc. up to 12.5 mm/hr
Each subroutine (IDL,Fortran,Matlab) performs the bit extraction and 
provides the user with the rain information.  Rain has less effect on C-band 
scatterometers than for Ku-band scatterometers, so the decision to remove 
rain affected data are left to the user.  

If a day is missing from the data set it is because all orbits for that day are missing.
Official information on gaps and the cause of missing data can be found at the podaac web
site listed above.

Orbit numbers can be related to day and month bytemaps by looking at the ASCAT_INFO.txt file.  
The date and time information for each rev# is provided in these files.
The columns represent:
Rev #, # of good WVC rows, EQCROSSDATE, EQCROSSTIME, EQCROSSLONG, ORBITPERIOD

The bytemap files are stored by year and month.  File names are ascat_YYYYMMDDv02.1.gz   Use any
gzip compatible zip utility to uncompress the data prior to using the read subroutines provided.



****************
**Orbit Files***
****************
Orbit files are also known as wind vector files.  They are located in the ascat_wind_vectors_v02.1 directory.  
Each file consists of one orbit of data.  The data are gridded into wind vector cells, 83 across the orbit 
and 3300 rows of cells along the orbit.

Reading subroutines are available in the ascat_orbit_support directory 
to assist in reading the orbit data files:
		get_ascat_orbit_v02.pro		IDL subroutine
		get_ascat_orbit_v02.f		Fortran subroutine
		get_ascat_orbit_v02.m		Matlab subroutine  

Some users have had trouble on other systems with different compilers.  If you are having
trouble reading the orbit files, check the specifics of how the file is opened and how the read
is performed to make sure it matches the requirements of your compiler.


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

	WINAL 	is the 10 meter ocean wind speed for the various ambiguities
	PHIAL 	is the wind direction for the various ambiguties (oceanographic convention)

	SOSAL	is the normalized rms after-the-fit residual of the observation minus 
	   	model sigma-nought.  Large SOSAL values indicate the observations did
		not fit the geophysical model function.
	We suggest discarding observations for which SOSAL.GT.1.9.

	WINDS 	is the smoothed version of the selected wind, WINAL(ISELECT)
	PHIWS 	is the smoothed version of the selected wind, PHIAL(ISELECT)

	WINGCM 	is the general circulation model wind speed used for nudging 
		(either NCEP or ECMWF)
	DIRGCM 	is the general circulation model wind direction used for 
		nudging (either NCEP or ECMWF)

	RAD_RAIN is the radiometer rain rates in mm/hr)
        	 RAD_RAIN=-999              no radiometer rain rate avaliable
   		 RAD_RAIN=0                 no rain measured
   		 RAD_RAIN=0.1               possible rain
   		 RAD_RAIN=0.2 through 12.5  definite rain and the given value 
					    is the rain rate in mm/hr
   		"no rain"         means no rain was detected within   +/- 50 km and 
				  +/- time given in MIN_DIFF.
   		"possible rain"   means some rain was detected within +/- 50 km and
				  +/- time given in MIN_DIFF.
   		"definite rain"   means rain was detected within      +/- 25 km and
				  +/- time given in MIN_DIFF.

	MIN_DIFF is the time difference in minutes between the scatterometer and 
		 the collocated radiometer.  A value of 255 means that no radiometer
		 observation was collocated with the scatterometer.

	NUDGE_NCEP	
	NUDGE_ECWMF
	Each data file contains two INTEGER(1) variables at the end of the file.
	Both variables supply information on which GCM was used for the nudging 
	field.  The first variable shows if NCEP data were used (0=used, 1=not used),
	and the second shows if ECMWF data were used (0=used,1=not used).  Since 
	GCM data are tri-linearly interpolated to the scatterometer data, it is possible
	that one of each map was used.  If both ECMWF and NCEP data were missing 
	for the required orbit, the orbit is not processed since no nudging field 
	would be available.

It is important to not use the data if the quality is suspect.  We recommend the 
following conditions be satisfied for the **very best** quality data.  You may choose otherwise:

	ICLASS(icel,iscan).ge. 2     (this would omit all outer swath data) 
	SOSAL(1,icel,iscan).le. 1.9  (yes, we mean the SOS of the lowest ambiguity)
	IRAIN_SCAT(icel,iscan).ne. 1
	MIN_DIFF(icel,iscan).le. 30 .and. RAD_RAIN(icel,iscan).le. 0.2


Due to the large number of files, we have subdivided them into directories of 
approximately 1000 files each.  01000to01999/		
				02000to02999/  etc.

The MetOp-A ASCAT dataset contains several gaps. The most recent missing orbit information is provided at  
ftp://podaac.jpl.nasa.gov/allData/ascat/preview/L2/metop_a/25km/README.datagap

Time can be associated with orbit number using the information in the time_orbit_number.txt
file included in this directory.



*****Contact Information*****
If you have questions about RSS Scatterometer data, contact RSS support:
http://www.remss.com/support
