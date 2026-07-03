from agent_platform.models.enums.dtype import DataType


def to_numpy_dtype(sample_type: DataType):
    match sample_type:
        case DataType.INT16:
            import numpy as np
            return np.int16
        case DataType.FLOAT32:
            import numpy as np
            return np.float32
        case DataType.INT8:
            import numpy as np
            return np.int8
        case DataType.UINT8:
            import numpy as np
            return np.uint8
        

def from_numpy_dtype(dtype) -> DataType:
    import numpy as np

    match dtype:
        case np.int16:
            return DataType.INT16
        case np.int8:
            return DataType.INT8
        case np.uint8:
            return DataType.UINT8
        case np.float32:
            return DataType.FLOAT32
        case np.float64:
            return DataType.FLOAT32
        case _:
            raise ValueError(f"Unsupported numpy dtype: {dtype}")