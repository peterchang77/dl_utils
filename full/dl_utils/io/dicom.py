import glob, numpy as np
import pydicom, mudicom

def load(path, ext='.dcm'):
    """
    Method to load DICOM object(s)

    """
    # --- Sort datasets
    datasets, fnames = load_datasets(path, ext=ext)

    # --- Extract arr and header
    data = load_pixel_array(datasets=datasets, fnames=fnames)
    meta = load_header(datasets=datasets)

    return data, meta 

def load_pixel_array(datasets, fnames):

    rows = datasets[0].Rows
    cols = datasets[0].Columns

    voxels = np.empty((rows, cols, len(datasets)), dtype='float32')

    for k in range(len(datasets)):

        slope = float(getattr(datasets[k], 'RescaleSlope', 1))
        intercept = float(getattr(datasets[k], 'RescaleIntercept', 0))
        voxels[..., k] = read_pixel_array(datasets[k], fnames[k]) * slope + intercept

    # --- Reshape (accounting for 4D acquisitions) 
    channels = extract_num_channels(datasets)
    voxels = voxels.reshape(rows, cols, -1, channels)
    voxels = np.moveaxis(voxels, 2, 0)

    return voxels.astype('int16')

def read_pixel_array(dataset, fname):
    """
    Method to attempt pixel array extraction via pydicom (uncompressed) or mudicom (compressed)

    """
    pixel_array = 0

    try:
        pixel_array = dataset.pixel_array
    except:
        pixel_array = mudicom.load(fname).image.numpy

    return pixel_array.astype('float32')

def load_header(path=None, datasets=None):

    if datasets is None:
        datasets, fnames = load_datasets(path, stop_before_pixels=True)

    spacing  = extract_spacing(datasets)

    meta = {}
    meta['header'] = extract_dicom_headers(datasets[0])
    meta['affine'] = calculate_affine(datasets, spacing)

    return meta 

def load_datasets(path, stop_before_pixels=False, ext='.dcm'):
    """
    Method to load all DICOM files in path

    """
    # --- Find fnames
    fnames = find_fnames(path, ext)

    # --- Load datasets
    datasets = [pydicom.read_file(f, force=True, stop_before_pixels=stop_before_pixels) for f in fnames]

    # --- Filter datasets (ensure presence of certain header fields)
    datasets, fnames = filter_datasets(datasets, fnames, stop_before_pixels=stop_before_pixels)

    # --- Sort by instance number (for 4D) then by position
    datasets, fnames = sort_slices_by_instance(datasets, fnames)
    datasets, fnames = sort_slices_by_position(datasets, fnames)

    return datasets, fnames

def find_fnames(path, ext):

    if type(path) is list:
        fnames = path

    else:
        if path[-4:] == ext:
            fnames = [path]

        else:
            if path[-1] != '/': path += '/'
            fnames = glob.glob('%s*%s' % (path, ext)) 
            assert len(fnames) > 0, 'Error no DICOM files found in path'

    return fnames 

def filter_datasets(datasets, fnames, stop_before_pixels):
    """
    Method to filter datasets by presence of fields

    """
    # --- Filter by required fields 
    REQ_FIELDS = ['ImageOrientationPatient', 'ImagePositionPatient', 'Rows', 'Columns', 'PixelSpacing']
    if not stop_before_pixels:
        REQ_FIELDS.append('PixelData')

    indices = [all([hasattr(d, r) for r in REQ_FIELDS]) for d in datasets]
    datasets = [d for i, d in zip(indices, datasets) if i] 
    fnames = [f for i, f in zip(indices, fnames) if i] 

    # --- Filter by common shape (Rows/Columns)
    f = [int(d.Rows) for d in datasets]
    datasets, fnames = filter_by_mode(f=f, datasets=datasets, fnames=fnames)

    f = [int(d.Columns) for d in datasets]
    datasets, fnames = filter_by_mode(f=f, datasets=datasets, fnames=fnames)

    # --- Filter by common ImageOrientationPatient
    f = [str(d.ImageOrientationPatient) for d in datasets]
    datasets, fnames = filter_by_mode(f=f, datasets=datasets, fnames=fnames)

    # --- Filter by SamplesPerPixel == 1
    ks = [n for n, d in enumerate(datasets) if d.SamplesPerPixel == 1]
    datasets = [datasets[k] for k in ks]
    fnames = [fnames[k] for k in ks]

    return datasets, fnames 

def filter_by_mode(f, datasets, fnames):
    """
    Method to filter datasets by the most common value in iterable f

    """
    mode = max(set(f), key=f.count)
    matches = [m == mode for m in f]
    datasets = [d for m, d in zip(matches, datasets) if m]
    fnames = [n for m, n in zip(matches, fnames) if m]

    return datasets, fnames

def calculate_affine(datasets, spacing):
    """
    Method to calculate ijk to xyz (real patient coordinates) affine transform

    """
    cos = extract_cosines(datasets)

    affine = np.identity(4, dtype=np.float32)

    affine[:3, 0] = (cos['row'] * spacing['col'])
    affine[:3, 1] = (cos['col'] * spacing['row'])
    affine[:3, 2] = (cos['slices'] * spacing['slices'])

    affine[:3, 3] = datasets[0].ImagePositionPatient

    # =================================================
    # Convert to ZYX instead of XYZ (default)
    # =================================================
    # 
    # ORIGINAL
    #
    #     [[x1, y1, z1, x4],  *  [[i],
    #      [x2, y2, z2, y4],      [j],
    #      [x3, y3, z3, z4],      [k],
    #      [0 , 0 , 0 , 1]]       [1]]
    # 
    # X = x1 * i + y1 * j + z1 * k + x4
    # Y = x2 * i + y2 * j + z2 * k + y4
    # Z = x3 * i + y3 * j + z3 * k + z4
    # 
    # CONVERTED
    #
    #     [[z3, y3, x3, z4],  *  [[k],
    #      [z2, y2, x2, y4],      [j],
    #      [z1, y1, x1, x4],      [i],
    #      [0,  0,  0,  1]]       [1]]
    # 
    # Z = z3 * k + y3 * j + x3 * i + z4
    # Y = z2 * k + y2 * j + x2 * i + y4
    # X = z1 * k + y1 * j + x1 * i + x4
    # 
    # =================================================

    affine[:3, :3] = affine[2::-1, 2::-1]
    affine[:3, 3] = affine[2::-1, 3]

    return affine 

def extract_spacing(datasets):

    spacing = {}

    # --- Extract xy-spacing
    spacing['row'] = float(datasets[0].PixelSpacing[0])
    spacing['col'] = float(datasets[0].PixelSpacing[1])

    # --- Extract z-spacing
    if len(datasets) > 1:
        spacing['slices'] = extract_slice_thickness(datasets) 

    else:
        spacing['slices'] = 0.

    return spacing

def extract_cosines(datasets):
    """
    Method extract direction cosines from DICOM image orientation vector

    Note that if direction cosines in ImageOrientationPatient header are not unit length,
    the values are normalized.

    Note that while row and column cosines are extracted from ImageOrientationPatient,
    the slice cosine is calculated from adjacent ImagePositionPatient headers because
    the z-axis is NOT guaranteed to be perpendicular to the XY-axis.

      cos['row'] = from ImageOrientationPatient
      cos['col'] = from ImageOrientationPatient
      cos['slices'] = from ImagePositionPatient

    IMPORTANT: it is assumed that datasets is sorted already by IPP

    """
    cos = {}

    dist = lambda x : np.sqrt(np.sum(x ** 2))
    norm = lambda x : x / dist(x)

    cos['row'] = norm(np.array(datasets[0].ImageOrientationPatient[:3]))
    cos['col'] = norm(np.array(datasets[0].ImageOrientationPatient[3:]))

    ipp = np.array([d.ImagePositionPatient for d in datasets])
    xyz = np.mean(ipp[1:] - ipp[:-1], axis=0)
    cos['slices'] = norm(xyz) 

    return cos 

def extract_slice_thickness(datasets):
    """
    Method to measure distance between adjacent slices based on ImagePositionPatient

    """
    cha = extract_num_channels(datasets)
    ipp = np.array([d.ImagePositionPatient for d in datasets])[::cha]
    xyz = np.mean(ipp[1:] - ipp[:-1], axis=0)

    return np.sqrt(np.sum(xyz ** 2)) 

def extract_slice_positions(datasets):
    """
    Method to extract slice positions by applying dot product of sli_cosine to 

    """
    cos = extract_cosines(datasets)
    slice_positions = [np.dot(cos['slices'], d.ImagePositionPatient) for d in datasets]

    return slice_positions 

def extract_num_channels(datasets):
    """
    Method to extract total number of channels by looking for repeat slice positions 

    """
    slice_positions = extract_slice_positions(datasets)
    n = len(slice_positions) 
    u = len(set(slice_positions))
    channels = int(n / u) if n % u == 0 else 1

    return channels

def sort_slices_by_instance(datasets, fnames):
    """
    Method to sort slices by instance number (if present)

    """
    if not hasattr(datasets[0], 'InstanceNumber'):
        return datasets, fnames

    slice_instances = [getattr(d, 'InstanceNumber', n) for n, d in enumerate(datasets)]

    datasets= [d for s, d in sorted(zip(slice_instances, datasets), key=lambda x:x[0])]
    fnames = [f for s, f in sorted(zip(slice_instances, fnames), key=lambda x:x[0])]

    return datasets, fnames

def sort_slices_by_position(datasets, fnames):
    """
    Method to sort slices by ImagePositionPatient 

    Note that this method sorts by IPP along the index (dimension) that spans the greatest distance 

    Importantly there no assumptions of any pre-sorting (e.g. via InstanceNumber), etc

    """
    ipp = np.array([d.ImagePositionPatient for d in datasets])
    ext = find_extreme_ipp(ipp)
    ind = np.argmax(np.abs(np.diff(ext, axis=0)))

    slice_positions = ipp[:, ind]

    datasets= [d for s, d in sorted(zip(slice_positions, datasets), key=lambda x:x[0])]
    fnames = [f for s, f in sorted(zip(slice_positions, fnames), key=lambda x:x[0])]

    return datasets, fnames

def find_extreme_ipp(ipp):
    """
    Method to find extreme values of ImagePositionPatient (assumed to be unsorted)

    """
    assert ipp.ndim == 2
    assert ipp.shape[1] == 3

    a = ipp.reshape(-1, 1, 3)
    b = ipp.reshape(1, -1, 3)

    dist = np.sum((a - b) ** 2, axis=2)
    i, j = np.unravel_index(np.argmax(dist), dist.shape)

    return np.stack((ipp[i], ipp[j]))

def extract_dicom_headers(dataset):

    DEFAULTS = {
        'PatientID': '',
        'StudyInstanceUID': None,
        'SeriesInstanceUID': None,
        'Modality': None, 
        'StudyDescription': '',
        'SeriesDescription': '',
        'StudyDate': '19000101',
        'AcquisitionTime': '000000',
        'AcquisitionNumber': None}

    d = {convert_to_snake_case(k): getattr(dataset, k, v) for k, v in DEFAULTS.items()}

    # --- MR attributes
    if hasattr(dataset, 'RepetitionTime'):

        DEFAULTS = {
            'RepetitionTime': None,
            'EchoTime': None,
            'EchoTrainLength': None,
            'EchoNumbers': None,
            'InversionTime': None,
            'MagneticFieldStrength': None,
            'PercentSampling': None,
            'FlipAngle': None}

        mr = {convert_to_snake_case(k): getattr(dataset, k, v) for k, v in DEFAULTS.items()}
        mr = {k: float(v) if ((v is not None) and (len(str(v)) != 0)) else None for k, v in mr.items()}

        d['_mr'] = mr

    # --- CT attributes
    if hasattr(dataset, 'KVP'):

        DEFAULTS = {
            'KVP': None, 
            'XRayTubeCurrent': None,
            'ExposureTime': None}

        ct = {convert_to_snake_case(k): getattr(dataset, k, v) for k, v in DEFAULTS.items()}
        ct = {k: float(v) if ((v is not None) and (len(str(v)) != 0)) else None for k, v in ct.items()}
        ct['convolutional_kernel'] = getattr(dataset, 'ConvolutionKernel', '')

        d['_ct'] = ct

    return d

def convert_to_snake_case(k):
    """
    Method to convert CamelCase to snake_case

    """
    KEY = {
        'PatientID': 'pid',
        'StudyInstanceUID': 'study_uid',
        'SeriesInstanceUID': 'series_uid',
        'Modality': 'modality', 
        'StudyDescription': 'study_description',
        'SeriesDescription': 'series_description', 
        'StudyDate': 'date',
        'AcquisitionTime': 'time',
        'AcquisitionNumber': 'acquisition_number',
        'RepetitionTime': 'TR',
        'EchoTime': 'TE',
        'EchoTrainLength': 'echo_train_length',
        'EchoNumbers': 'echo_numbers',
        'InversionTime': 'inversion_time',
        'MagneticFieldStrength': 'magnetic_field_strength',
        'PercentSampling': 'percent_sampling',
        'FlipAngle': 'flip_angle',
        'KVP': 'kVp', 
        'XRayTubeCurrent': 'mA',
        'ExposureTime': 'mAs',
        'ConvolutionKernel': 'convolution_kernel'}

    return KEY[k] if k in KEY else k

if __name__ == '__main__':

    # path = '/mnt/hdd0/data/raw/aspects/v1/anon/uci/1.2.840.113704.1025835418667756012145209405977055349941561850811/1.2.840.113704.194860334756773965156495730863715113111262594171/'
    # path = '/mnt/hdd0/data/raw/aspects/v1/anon/uci/1.2.840.113704.1058263210463269322780517880524524352027452969473/1.2.840.113704.469535596362113471059201116848635401193047824426/'
    # path = '/mnt/hdd0/data/raw/lvo/ko/lvoko/CT_CTA_Head/DE_CarotidAngio__0.75__D26f__#PP__M_0.6/'
    # arr, meta = load(path=path)

    pass
