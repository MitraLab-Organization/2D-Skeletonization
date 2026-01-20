import numpy as np

def check_connectivity(conn):
    if isinstance(conn, int):
        if conn == 4:
            return np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=int)
        elif conn == 8:
            return np.ones((3, 3), dtype=int)
        elif conn == 6:
            c = np.zeros((3, 3, 3), dtype=int)
            c[1, 1, 1] = 1
            c[0, 1, 1] = 1
            c[2, 1, 1] = 1
            c[1, 0, 1] = 1
            c[1, 2, 1] = 1
            c[1, 1, 0] = 1
            c[1, 1, 2] = 1
            return c
        elif conn == 18:
            slice1 = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
            slice2 = np.ones((3, 3))
            slice3 = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
            return np.stack([slice1, slice2, slice3], axis=2)
        elif conn == 26:
            return np.ones((3, 3, 3), dtype=int)
        else:
            raise ValueError(f"Invalid connectivity value: {conn}")
    elif isinstance(conn, np.ndarray):
        if conn.ndim > 3:
            raise ValueError("Connectivity must be 2-D or 3-D.")
        if any(s % 2 == 0 for s in conn.shape):
            raise ValueError("The connectivity size must be odd.")
        if not np.array_equal(conn, np.flip(conn)):
            raise ValueError("The connectivity must be symmetric through its center element.")
        return conn
    else:
        raise TypeError("conn must be an integer or a numpy array.")