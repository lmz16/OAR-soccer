import json
import os

def jsoncat(dir_name, split):
    alljson = {}
    for fn in os.listdir(dir_name):
        if "{0}_homography_".format(split) in fn:
            with open("{0}/{1}".format(dir_name, fn)) as f:
                alljson.update(json.load(f))
    with open("{0}/{1}_homography.json".format(dir_name, split), "w") as f:
        json.dump(alljson, f)

if __name__ == "__main__":
    jsoncat("4039", 'first')
    jsoncat("4039", 'second')