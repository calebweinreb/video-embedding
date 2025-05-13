"""Methods for visualization of videos and video embeddings."""

def play_videos(videos, rows, cols, inches=3):
    """
    Play multiple videos in a grid with specified rows and columns.
    
    videos: List of videos to display, each shaped as (frames, height, width, channels).
    rows: Number of rows in the grid.
    cols: Number of columns in the grid.
    """
    num_videos = len(videos)
    
    if rows * cols < num_videos:
        raise ValueError("Grid size (rows * cols) is smaller than the number of videos.")
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols * inches, rows * inches))
    
    if rows == 1:
        axes = axes.flatten() if num_videos > 1 else [axes]
    elif cols == 1:
        axes = axes.flatten()
    else:
        axes = axes.flatten()

    ims = []
    for i, video in enumerate(videos):
        axes[i].imshow(video[0, :, :, :])
        axes[i].axis('off')
        im = axes[i].imshow(video[0, :, :, :], animated=True)
        ims.append(im)
    
    for i in range(num_videos, rows * cols):
        axes[i].axis('off')
    
    plt.close(fig)  

    def init():
        for i, video in enumerate(videos):
            ims[i].set_data(video[0, :, :, :])
        return ims

    def animate(frame):
        for i, video in enumerate(videos):
            ims[i].set_data(video[frame, :, :, :])
        return ims

    num_frames = min(video.shape[0] for video in videos)
    anim = animation.FuncAnimation(fig, animate, init_func=init, frames=num_frames, interval=50, blit=True)
    
    return HTML(anim.to_html5_video())