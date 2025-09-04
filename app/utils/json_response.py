# utils/json_response.py
import json
import numpy as np
import pandas as pd
from fastapi.responses import JSONResponse


class CustomJSONResponse(JSONResponse):
    """
    A universal JSONResponse that knows how to serialize
    numpy and pandas scalar types + timestamps.
    """

    def render(self, content: any) -> bytes:
        def default(o):
            # numpy scalars
            if isinstance(o, (np.integer,)):
                return int(o)
            if isinstance(o, (np.floating,)):
                return float(o)
            if isinstance(o, (np.bool_,)):
                return bool(o)

            # pandas timestamps/timedeltas
            if isinstance(o, (pd.Timestamp,)):
                return o.isoformat()
            if isinstance(o, (pd.Timedelta,)):
                return str(o)

            # Fallback to let the default encoder raise a TypeError for unknown types
            return json.JSONEncoder.default(o)

        return json.dumps(content, default=default, ensure_ascii=False).encode("utf-8")
