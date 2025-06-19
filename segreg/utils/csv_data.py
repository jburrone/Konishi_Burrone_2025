from dataclasses import dataclass


@dataclass
class CSVData:
    mov_imgfile: str
    mov_segfile: str
    ref_imgfile: str
    ref_segfile: str
    ref_scale_var: float
    left_bound: int
    top_bound: int
    width: int
    height: int
    min_z: int
    max_z: int
    z_step: int
    tz_center: float


@dataclass
class CSVData_xyzTz:
    z_segreg: int
    y_offmov: int
    y_segreg: int
    x_offmov: int
    x_segreg: int
    tz_segreg: float
    sx_segreg: float
    sy_segreg: float
    mi: float
    opixmatch: float
    jaccard: float
    omatch: float
    ce: float
    all: float


@dataclass
class CSVData_Ty:
    ty_segreg: float
    y_segreg: int
    x_segreg: int
    sx_segreg: float
    sy_segreg: float
    mi: float
    opixmatch: float
    jaccard: float
    omatch: float
    ce: float
    all: float


@dataclass
class CSVData_Tx:
    tx_segreg: float
    y_segreg: int
    x_segreg: int
    sx_segreg: float
    sy_segreg: float
    mi: float
    opixmatch: float
    jaccard: float
    omatch: float
    ce: float
    all: float
