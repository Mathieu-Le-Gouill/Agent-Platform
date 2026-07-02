from agent_platform.models.enums.dtype import AudioSampleType


def to_numpy_dtype(sample_type: AudioSampleType):
    match sample_type:
        case AudioSampleType.INT16:
            import numpy as np
            return np.int16
        case AudioSampleType.FLOAT32:
            import numpy as np
            return np.float32
        case AudioSampleType.INT8:
            import numpy as np
            return np.int8
        case AudioSampleType.UINT8:
            import numpy as np
            return np.uint8
        

def from_numpy_dtype(dtype) -> AudioSampleType:
    import numpy as np

    match dtype:
        case np.int16:
            return AudioSampleType.INT16
        case np.int8:
            return AudioSampleType.INT8
        case np.uint8:
            return AudioSampleType.UINT8
        case np.float32:
            return AudioSampleType.FLOAT32
        case np.float64:
            return AudioSampleType.FLOAT32
        case _:
            raise ValueError(f"Unsupported numpy dtype: {dtype}")