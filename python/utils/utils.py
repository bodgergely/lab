

def writeFrame(frameFile, frame):
    import scipy.misc
    scipy.misc.toimage(frame, channel_axis=2).save(frameFile + ".jpg")