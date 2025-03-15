## Download NODDI, Oddball, & CN-EPFL datasets
### NODDI
Visit https://osf.io/94c5t/files/osfstorage and download EEG1.zip, EEG2.zip, and fMRI.zip.  
Unzip all files and merge EEG1 and EEG2 into one (e.g., EEG). Both EEG and fMRI directories should have 16 individuals.  
```bash
NODDI
└── EEG
    └── 32
    └── 35
    ...
└── fMRI
    └── 32
    └── 35
    ...
```

### Oddball
Visit https://legacy.openfmri.org/dataset/ds000116/ and download processed data for 17 subjects.
```bash
http://openfmri.s3.amazonaws.com/tarballs/ds116_sub001.tgz
http://openfmri.s3.amazonaws.com/tarballs/ds116_sub002.tgz
http://openfmri.s3.amazonaws.com/tarballs/ds116_sub003.tgz
...
http://openfmri.s3.amazonaws.com/tarballs/ds116_sub017.tgz
```

Unzip all files with command
```bash
tar -xvzf /path/to/yourfile.tgz
```
The Oddball data should have the following structure:
```bash
Oddball
└── sub001
└── sub002
    ...
└── sub017
```

### CN-EPFL
Visit https://openneuro.org/datasets/ds002158/versions/1.0.2/download and follow the download instructions.  
The CN-EPFL data includes 20 individuals and should have the following structure:
```bash
CN-EPFL
└── ds002158-download
    └── sub-02
    └── sub-04
        ...
    └── sub-26
```

## Process and create EEG-fMRI pairs
Modify the raw dataset directories (`raw_data_roots`) and processed dataset directories (`processed_data_roots`) (i.e., where to save h5 files) in [data_cfg.py](../data_cfg.py).  
An example of processing NODDI dataset is below
```bash
# data_name includes 'NODDI', 'Oddball', 'CNEPFL'
python process_data.py --data_name NODDI
```