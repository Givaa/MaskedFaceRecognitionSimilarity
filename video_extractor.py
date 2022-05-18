from os import listdir, path
import re

directory = "video"
destinationDirectory = "video_final"

folders = list(filter(lambda file: file.find("_0") >0, listdir(directory)))
for i, f in enumerate(folders):
    folders[i] = f.split("_0")[0]

def getVideos(path):
    return list(filter(lambda file: re.match("_\d_\d_1\.", file), listdir(path)))

def frameExtractor(path, destinationFolder):
    print("COSA")

for identity in folders:
    videosUnmasked = getVideos(path.join(directory, identity+"_0"))
    videosMasked = getVideos(path.join(directory, identity+"_1"))
    for v in videosUnmasked:
        video = path.join(directory, identity+"_0", v)
        videoType = re.math(r'\d{3}_(\d)_')
        frameExtractor(video, path.join(destinationDirectory, identity))
        

a