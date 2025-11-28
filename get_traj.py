import h5py
import numpy as np

def get_traj_length_stats(hdf5_path):
    f = h5py.File(hdf5_path, "r")

    # playback_dataset.py랑 같은 방식으로 demo 리스트 정렬
    demos = list(f["data"].keys())
    inds = np.argsort([int(elem[5:]) for elem in demos])  # 'demo_0', 'demo_1' 이런 형식 가정
    demos = [demos[i] for i in inds]

    lengths = []

    for ep in demos:
        actions = f[f"data/{ep}/actions"][()]  # shape = (T, act_dim)
        T = actions.shape[0]
        lengths.append(T)

    f.close()

    lengths = np.array(lengths)
    print(f"# episodes          : {len(lengths)}")
    print(f"min traj length     : {lengths.min()}")
    print(f"max traj length     : {lengths.max()}")
    print(f"mean traj length    : {lengths.mean():.2f}")
    print(f"median traj length  : {np.median(lengths):.2f}")
    print(f"25% / 75% quantiles : {np.percentile(lengths, [25, 75])}")

    return lengths

if __name__ == "__main__":
    hdf5_path = "/home/robros/git/diffusion-qcfql/robomimic/dataset/transport/mh/low_dim_v15.hdf5"
    lengths = get_traj_length_stats(hdf5_path)

    hdf5_path2 = "/home/robros/git/diffusion-qcfql/robomimic/dataset/transport/ph/low_dim_v15.hdf5"
    lengths2 = get_traj_length_stats(hdf5_path)
