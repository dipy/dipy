import dicom

from enthought.mayavi import mlab

from glob import glob

import dipy.io.bmatrix as b

import struct

datadir="/home/ian/Data/20100114_195840/Series_012_CBU_DTI_64D_1A/"

dcmfiles=glob(datadir+"*.dcm")

filename=dcmfiles[1]

data=dicom.read_file(filename)

#(0029, 1010) [CSA Image Header Info]             OB: Array of 11884 bytes
csainfo = data[0x29,0x1010] 

csadic={'filename': filename}

(csadic['hdr_id'], csadic['no_tags'], csadic['check']) = struct.unpack('4s4x2I',csainfo[0:16])

csadic['tags'] = {}

ptr = 16

maxptr = 11568

for tag in range(csadic['no_tags']):
#for tag in range(6)
    name, vm, vr, syngodt, n_items, xx = struct.unpack('64sI4sIII',csainfo[ptr:ptr+84])
    nullpos=name.find(chr(0))
    if nullpos < 0:
        nullpos=64
    name = name[0:nullpos]
    csadic['tags'][name]={'pointer': ptr, 'n_items': n_items}
    ptr+=84
    for item in range(n_items):
        csadic['tags'][name]['pointer'] = ptr
        x0,x1,x2,x3 = struct.unpack('IIII2x',csainfo[ptr:ptr+18])
        csadic['tags'][name]['xx'] = [x0,x1,x2,x3]
        ptr+=18
#        item_len = min(x1,maxptr-ptr)
#        format = item_len*'b'
#        csadic['tags'][name]['value'] = struct.unpack(format,csainfo[ptr:ptr+item_len])
#        ptr+=item_len
#    csadic['tags'][name]['end_pointer'] = ptr     

print 'csadic', csadic

#print label, number_of_frames

#B_value, B_vec, G_direction, B_value_B_matrix, no_mosaic=b.loadbinfodcm(filename,spm_converted=1)

'''
voldata=data.pixel_array.reshape(128,128,49)

v4=voldata.reshape(7,128,7,128)
v4p=np.rollaxis(v4,2,1)
v3=v4p.reshape(49,128,128)

imshow(v3[22,:,:])

'''


'''
(0008, 0005) Specific Character Set              CS: 'ISO_IR 100'
(0008, 0008) Image Type                          CS: ['ORIGINAL', 'PRIMARY', 'DIFFUSION', 'NONE', 'ND', 'MOSAIC']
(0008, 0012) Instance Creation Date              DA: '20100114'
(0008, 0013) Instance Creation Time              TM: '203419.375000'
(0008, 0016) SOP Class UID                       UI: MR Image Storage
(0008, 0018) SOP Instance UID                    UI: 1.3.12.2.1107.5.2.32.35119.201001142034191872808941
(0008, 0020) Study Date                          DA: '20100114'
(0008, 0021) Series Date                         DA: '20100114'
(0008, 0022) Acquisition Date                    DA: '20100114'
(0008, 0023) Content Date                        DA: '20100114'
(0008, 0030) Study Time                          TM: '195840.718000'
(0008, 0031) Series Time                         TM: '203001.890000'
(0008, 0032) Acquisition Time                    TM: '203418.360000'
(0008, 0033) Content Time                        TM: '203419.375000'
(0008, 0050) Accession Number                    SH: ''
(0008, 0060) Modality                            CS: 'MR'
(0008, 0070) Manufacturer                        LO: 'SIEMENS'
(0008, 0080) Institution Name                    LO: 'MRC-CBU'
(0008, 0081) Institution Address                 ST: 'Chaucer Road  15,Cambridge,UK,GB,CB2 2EF'
(0008, 0090) Referring Physician's Name          PN: ''
(0008, 1010) Station Name                        SH: 'MRC35119'
(0008, 1030) Study Description                   LO: 'CBU^Neuroimaging'
(0008, 103e) Series Description                  LO: 'CBU_DTI_64D_1A'
(0008, 1050) Performing Physician's Name         PN: ''
(0008, 1070) Operators' Name                     PN: 'MC'
(0008, 1090) Manufacturer's Model Name           LO: 'TrioTim'
(0008, 1140)  Referenced Image Sequence   3 item(s) ---- 
   (0008, 1150) Referenced SOP Class UID            UI: MR Image Storage
   (0008, 1155) Referenced SOP Instance UID         UI: 1.3.12.2.1107.5.2.32.35119.2010011420070434054586384
   ---------
   (0008, 1150) Referenced SOP Class UID            UI: MR Image Storage
   (0008, 1155) Referenced SOP Instance UID         UI: 1.3.12.2.1107.5.2.32.35119.2010011420070721803086388
   ---------
   (0008, 1150) Referenced SOP Class UID            UI: MR Image Storage
   (0008, 1155) Referenced SOP Instance UID         UI: 1.3.12.2.1107.5.2.32.35119.201001142007109937386392
   ---------
(0010, 0010) Patient's Name                      PN: '14012010'
(0010, 0020) Patient ID                          LO: 'METHODS'
(0010, 0030) Patient's Birth Date                DA: '19770222'
(0010, 0040) Patient's Sex                       CS: 'M'
(0010, 1010) Patient's Age                       AS: '032Y'
(0010, 1030) Patient's Weight                    DS: '88.000000'
(0018, 0020) Scanning Sequence                   CS: 'EP'
(0018, 0021) Sequence Variant                    CS: ['SK', 'SP']
(0018, 0022) Scan Options                        CS: ['PFP', 'FS']
(0018, 0023) MR Acquisition Type                 CS: '2D'
(0018, 0024) Sequence Name                       SH: 'ep_b1000#39'
(0018, 0025) Angio Flag                          CS: 'N'
(0018, 0050) Slice Thickness                     DS: '2.500000'
(0018, 0080) Repetition Time                     DS: '6600.000000'
(0018, 0081) Echo Time                           DS: '93.000000'
(0018, 0083) Number of Averages                  DS: '1.000000'
(0018, 0084) Imaging Frequency                   DS: '123.251815'
(0018, 0085) Imaged Nucleus                      SH: '1H'
(0018, 0086) Echo Number(s)                      IS: '1'
(0018, 0087) Magnetic Field Strength             DS: '3.000000'
(0018, 0088) Spacing Between Slices              DS: '3.000000'
(0018, 0089) Number of Phase Encoding Steps      IS: '102'
(0018, 0091) Echo Train Length                   IS: '1'
(0018, 0093) Percent Sampling                    DS: '100.000000'
(0018, 0094) Percent Phase Field of View         DS: '100.000000'
(0018, 0095) Pixel Bandwidth                     DS: '1395.000000'
(0018, 1000) Device Serial Number                LO: '35119'
(0018, 1020) Software Version(s)                 LO: 'syngo MR B17'
(0018, 1030) Protocol Name                       LO: 'CBU_DTI_64D_1A'
(0018, 1251) Transmit Coil Name                  SH: 'Body'
(0018, 1310) Acquisition Matrix                  US: [128, 0, 0, 128]
(0018, 1312) In-plane Phase Encoding Direction   CS: 'COL'
(0018, 1314) Flip Angle                          DS: '90.000000'
(0018, 1315) Variable Flip Angle Flag            CS: 'N'
(0018, 1316) SAR                                 DS: '0.421666'
(0018, 1318) dB/dt                               DS: '0.000000'
(0018, 5100) Patient Position                    CS: 'HFS'
(0019, 0010) Private Creator                     OB: Array of 18 bytes
(0019, 1008) Private tag data                    OB: 'IMAGE NUM 4 '
(0019, 1009) Private tag data                    OB: '1.0 '
(0019, 100a) Private tag data                    OB: '0\x00'
(0019, 100b) Private tag data                    OB: '40'
(0019, 100c) Private tag data                    OB: '1000'
(0019, 100d) Private tag data                    OB: 'DIRECTIONAL '
(0019, 100e) Private tag data                    OB: Array of 24 bytes
(0019, 100f) Private tag data                    OB: 'Fast* '
(0019, 1011) Private tag data                    OB: 'No'
(0019, 1012) Private tag data                    OB: '\x00\x00\x00\x00\x00\x00\x00\x00\x1c\xfb\xff\xff'
(0019, 1013) Private tag data                    OB: '\x00\x00\x00\x00\x00\x00\x00\x00\x1c\xfb\xff\xff'
(0019, 1014) Private tag data                    OB: '0\\0\\0 '
(0019, 1015) Private tag data                    OB: Array of 24 bytes
(0019, 1016) Private tag data                    OB: '271.5475'
(0019, 1017) Private tag data                    OB: '1 '
(0019, 1018) Private tag data                    OB: '2800'
(0019, 1027) Private tag data                    OB: Array of 48 bytes
(0019, 1028) Private tag data                    OB: '\xaeG\xe1z\x14\x0e3@'
(0019, 1029) Private tag data                    OB: Array of 384 bytes
(0020, 000d) Study Instance UID                  UI: 1.3.12.2.1107.5.2.32.35119.30000010011408520750000000022
(0020, 000e) Series Instance UID                 UI: 1.3.12.2.1107.5.2.32.35119.2010011420292594820699190.0.0.0
(0020, 0010) Study ID                            SH: '1'
(0020, 0011) Series Number                       IS: '12'
(0020, 0012) Acquisition Number                  IS: '40'
(0020, 0013) Instance Number                     IS: '40'
(0020, 0032) Image Position (Patient)            DS: ['-805.000000', '-825.019119', '-75.097641']
(0020, 0037) Image Orientation (Patient)         DS: ['1.000000', '0.000000', '0.000000', '0.000000', '0.999986', '-0.005236']
(0020, 0052) Frame of Reference UID              UI: 1.3.12.2.1107.5.2.32.35119.1.20100114195840906.0.0.0
(0020, 1040) Position Reference Indicator        LO: ''
(0020, 1041) Slice Location                      DS: '-79.416382'
(0028, 0002) Samples per Pixel                   US: 1
(0028, 0004) Photometric Interpretation          CS: 'MONOCHROME2'
(0028, 0010) Rows                                US: 896
(0028, 0011) Columns                             US: 896
(0028, 0030) Pixel Spacing                       DS: ['1.796875', '1.796875']
(0028, 0034) Pixel Aspect Ratio                  IS: ['1', '1']
(0028, 0100) Bits Allocated                      US: 16
(0028, 0101) Bits Stored                         US: 12
(0028, 0102) High Bit                            US: 11
(0028, 0103) Pixel Representation                US: 0
(0028, 0106) Smallest Image Pixel Value          US or SS: '\x00\x00'
(0028, 0107) Largest Image Pixel Value           US or SS: '\x08\x03'
(0028, 1050) Window Center                       DS: '171.000000'
(0028, 1051) Window Width                        DS: '401.000000'
(0028, 1055) Window Center & Width Explanation   LO: 'Algo1'
(0029, 0010) Private Creator                     OB: Array of 18 bytes
(0029, 0011) Private Creator                     OB: Array of 22 bytes
(0029, 1008) [CSA Image Header Type]             OB: 'IMAGE NUM 4 '
(0029, 1009) [CSA Image Header Version]          OB: '20100114'
(0029, 1010) [CSA Image Header Info]             OB: Array of 11888 bytes
(0029, 1018) [CSA Series Header Type]            OB: 'MR'
(0029, 1019) [CSA Series Header Version]         OB: '20100114'
(0029, 1020) [CSA Series Header Info]            OB: Array of 80248 bytes
(0029, 1160) [Series Workflow Status]            OB: 'com '
(0032, 1060) Requested Procedure Description     LO: 'CBU Neuroimaging'
(0040, 0244) Performed Procedure Step Start Date DA: '20100114'
(0040, 0245) Performed Procedure Step Start Time TM: '195840.828000'
(0040, 0253) Performed Procedure Step ID         SH: 'MR20100114195840'
(0040, 0254) Performed Procedure Step Descriptio LO: 'CBU^Neuroimaging'
(0051, 0010) Private Creator                     OB: Array of 18 bytes
(0051, 1008) Private tag data                    OB: 'IMAGE NUM 4 '
(0051, 1009) Private tag data                    OB: '1.0 '
(0051, 100a) Private tag data                    OB: 'TA 00.04'
(0051, 100b) Private tag data                    OB: '128p*128'
(0051, 100c) Private tag data                    OB: 'FoV 1610*1610 '
(0051, 100e) Private tag data                    OB: 'Tra>Cor(-0.3) '
(0051, 100f) Private tag data                    OB: 'T:HEA;HEP '
(0051, 1011) Private tag data                    OB: 'p2'
(0051, 1012) Private tag data                    OB: 'TP 0'
(0051, 1013) Private tag data                    OB: '+LPH'
(0051, 1015) Private tag data                    OB: 'R '
(0051, 1016) Private tag data                    OB: Array of 28 bytes
(0051, 1017) Private tag data                    OB: 'SL 2.5'
(0051, 1019) Private tag data                    OB: 'A1/PFP/FS '
(7fe0, 0010) Pixel Data                          OW/OB: Array of 1605632 bytes
'''