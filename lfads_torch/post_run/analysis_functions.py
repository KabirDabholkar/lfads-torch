import numpy as np

def participation_ratio(data: np.ndarray):
    assert len(data.shape)==2
    s = np.linalg.svd(data,compute_uv=False)
    return ((s**2).sum()**2)/(s**4).sum()

if __name__ == '__main__':
    print(
        participation_ratio(
            np.random.uniform(size=(200,100))-0.5
            # np.random.normal(size=(200, 100))
        )
    )