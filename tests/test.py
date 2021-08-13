import pylampld
from admix.data import read_int_mat
from admix.plot import plot_local_anc
import numpy as np
from os.path import join
import matplotlib.pyplot as plt

def test_basic():
    assert True

# def test_2():
#     data_dir = "/Users/kangchenghou/work/LAMP-LD/test_data/ten_region/"
#
#     pos = np.loadtxt(join(data_dir, "pos.txt"))
#     admix_hap = read_int_mat(join(data_dir, "admix.hap"))
#     ref_haps = [
#         read_int_mat(join(data_dir, "EUR.hap")),
#         read_int_mat(join(data_dir, "AFR.hap"))
#     ]
#
#     model = pylampld.LampLD(n_snp = len(pos), n_anc = len(ref_haps), n_proto=4, window_size=300)
#     model.set_pos(pos)
#
#     model.fit(ref_haps)
#
#     inferred_lanc = model.infer_lanc(admix_hap)
#     gt_lanc = read_int_mat(join(data_dir, "admix.lanc"))
#
#     print("Accuracy: {np.mean(inferred_lanc == gt_lanc):.2f}")
#
#     plot_local_anc(inferred_lanc[0:10, :])
#     plt.title("Inferred")
#     plt.show()
#     plot_local_anc(gt_lanc[0:10, :])
#     plt.title("Groundtruth")
#     plt.show()
