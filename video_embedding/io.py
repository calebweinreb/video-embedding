"""Video reader for self-supervised learning."""
def get_clip(path, start, duration=60):
    """."""
    reader = OpenCVReader(path)
    clip = reader[start : start + duration]
    return np.stack(clip)


video_paths = [f for f in glob.glob('../../data/*/*.cropped.mp4') if not 'down2' in f]
lengths = [len(OpenCVReader(p)) for p in video_paths]