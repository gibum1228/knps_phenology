import phenoKOR as pk

if __name__ == "__main__":
    root = "/Users/beom/Desktop/git/data/knps/"

    mask = pk.load_mask(root, "roi.mat")
    df, imgs = pk.load_image(root + "sungsamjae/2020")
    rois = pk.load_roi(imgs, mask)

    r_cc, g_cc, b_cc = pk.get_cc(rois)

    pk.show_plot(g_cc)

    print(mask.shape) # (4000, 6000)