import phenoKOR as pk
import matplotlib.pyplot as plt

if __name__ == "__main__":
    root = "/Users/beom/Desktop/git/data/knps/"

    print("load mask()")
    mask = pk.load_mask(root, "roi.mat")
    print("load image()")
    df, imgs = pk.load_image(root + "sungsamjae/2020/sample")
    print("load_roi()")
    rois = pk.load_roi(imgs, mask)

    print("get cc()")
    r_cc, g_cc, b_cc = pk.get_cc(rois)

    print(g_cc)

    plt.plot(g_cc)
    plt.show()

    print(mask.shape) # (4000, 6000)