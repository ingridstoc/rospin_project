FUSAR-Ship Dataset version 1.0
copyright at Fudan University March 2020

1. Introcution
FUSAR-Ship is an open SAR-AIS matchup dataset of Gaofen-3 satellite propared by the Key Lab of Information Science of Electromagnetic Waves (MoE) of Fudan University. Gaofen-3 (GF-3) is China’s first civil C-Band fully polarimetric spaceborne synthetic aperture radar (SAR) primarily missioned for ocean remote sensing and marine monitoring. The FUSAR-Ship dataset is constructed by the proposed automatic SAR-AIS matchup procedure on over 100 GF-3 scenes covering a large variety of sea, land, coast, river and island scenarios. It includs over 5000 ship image chips with AIS messages (AIS: automatic identification system) and some other types of marine targets and background clutters. FUSAR-Ship is intended as an open benchmark dataset for ship and marine target detection and recognition.

2. Structure
- All image chips are 512x512 boxes cropped from the original GF-3 L1A images. The ship is always located at the center, but the chip may include nearby ships or other objects. These image chips are stored under subfolder named after its 'category/subcategory'. The filename of each sample follows the convention: Ship_CxxSyyNzzzz.tiff, where xx is the index of category, yy is the index of subcategory and zzzz is the index of this particular sample.
- The matchup meta data is compiled in the file 'meta.csv' or 'meta.xls', which follow the format of:
id	mmsi	length	width	polarMode	centerLookAngle	heightspace	widthspace	path
Ship_C01S01N0001	477939300	138	26	DH	42.51	1.702183	1.124222	Cargo\AggregatesCarrier
Ship_C01S01N0002	371469000	87	12	DH	42.51	1.701591	1.124222	Cargo\AggregatesCarrier
Ship_C01S02N0001	351445000	229	32	DH	27.28	1.72615	1.124222	Cargo\BulkCarrier
...
Definition of each field is explained as follows:
id - file name
mmsi - ship MMSI extracted from AIS
length, width - ship size extracted from AIS
polarMode - GF-3 polarization mode
centerLookAngle - GF-3 incident angle at the scene center
heightspace, widthspace - GF-3 SAR image pixel size
path - path to the file

3. Citation
Please cite the following paper if you use FUSAR-Ship in your publications:
[Hou et al. FUSAR-Ship: a High-resolution SAR-AIS Matchup Dataset of Gaofen-3 for Ship Detection and Recognition, SCIENCE CHINA Information Sciences, 2020]