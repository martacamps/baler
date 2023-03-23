import numpy as np
import os


def test_CMS():
    print(os.getcwd())
    input_path = "./data/example_CMS/example_CMS.npz"
    project_path = "./projects/example_CMS/"
    loaded = np.load(input_path)
    names = loaded["names"]
    before = np.transpose(np.load(input_path)["data"])
    after = np.transpose(
        np.load(project_path + "/decompressed_output/decompressed.npz")["data"]
    )
    residual = after - before

    new_average_list = []
    new_rms_list = []
    for i in range(0, len(names)):
        new_rms_list.append(np.sqrt(np.mean(np.square(residual[i]))))
        new_average_list.append(np.mean(residual[i]))
    new_average = np.array(new_average_list)
    new_rms = np.array(new_rms_list)

    np.savez(
        "./tests/new_CMS_performance.npz",
        average=np.array(new_average_list),
        rms=np.array(new_rms_list),
    )

    old_loaded = np.load("./tests/old_CMS_performance.npz")
    old_average = old_loaded["average"]
    old_rms = old_loaded["rms"]

    assert np.isclose(new_average, old_average, rtol=1e-05).all()
    assert np.isclose(new_rms, old_rms, rtol=1e-05).all()
