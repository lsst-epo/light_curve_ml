This directory contains csv text dumps of the macho photometry data, organised by field and tile.

There is one directory per field, each containing one csv file per tile.

All files are gzip compressed.

Field-centre coordinates are given at: http://macho.nci.org.au/Macho_fields.html
Star coordinates, per field, are available at: http://macho.nci.org.au/macho_stars/
Use this metadata to determine which files are of interest to you.

For enquiries, contact: macho@anusf.anu.edu.au

CSV File Format:
----------------

Each CSV file contains one line per datapoint, with 39 columns per line - the field separator on each line is the semicolon ";".

The fields in order are:

1.  <blank>
2.  field id
3.  tile id
4.  star sequence id
5.  observation date (Modified Julian Date: MJD = JD - 2400000.5)
6.  observation id
7.  side of pier
8.  exposure time
9.  airmass
10. red magnitude
11. red error
12. red normsky
13. red type
14. red crowd
15. red chi2
16. red mpix
17. red cosmicrf
18. red amp
19. red xpix
20. red ypix
21. red avesky
22. red fwhm
23. red tobs
24. red cut
25. blue magnitude
26. blue error
27. blue normsky
28. blue type
29. blue crowd
30. blue chi2
31. blue mpix
32. blue cosmicrf
33. blue amp
34. blue xpix
35. blue ypix
36. blue avesky
37. blue fwhm
38. blue tobs
39. blue cut


Classifications
Number. Acronym - Description
1. RRL AB - RRLyrae ab
2. RRL C - RRLyrae c
3. RRL E - RRLyrae e
4. Ceph Fundamental
5. Ceph 1st Overtone
6. LPV WoodA
7. LPV WoodB
8. LPV WoodC
9. LPV WoodD
10. EB - Eclipsing Binary
11. RRL + GB - RR Lyrae and GB blends
