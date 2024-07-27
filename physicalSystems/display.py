import multiprocessing
from mayavi import mlab
import numpy as np


def render_box_and_particles(box_size, particles, *, filename=None, frame=None):
    print('\\'.join(__file__.split('\\')[:-1]))
    scales = 10 / box_size
    # Rescale Box dimension
    [box_length, box_width, box_height] = scales * box_size
    # get particles positions moment and easy axes direction
    positions, moments, easy_axes = particles
    positions *= scales  # Rescale positions
    num_particles = positions.shape[0]

    # Create a new 3D figure
    mlab.figure(size=(800, 800))

    # Render the box (outline)
    box = mlab.points3d(
        [0, box_length, box_length, 0, 0, box_length, box_length, 0],
        [0, 0, box_width, box_width, 0, 0, box_width, box_width],
        [0, 0, 0, 0, box_height, box_height, box_height, box_height],
        mode="cube",
        color=(0, 0, 0),
        scale_factor=1,
        opacity=0.1
    )

    # Render the particles
    particles_scat = mlab.points3d(
        positions[:, 0], positions[:, 1], positions[:, 2],
        color=(0, 1, 0),
        scale_factor=.5
    )

    # Render the arrows
    for i in range(num_particles):
        # draw moments
        origin = positions[i]
        direction = moments[i] / np.linalg.norm(moments[i])  # Normalized direction
        mlab.quiver3d(
            origin[0], origin[1], origin[2],
            direction[0], direction[1], direction[2],
            color=(1, 0, 0),
            scale_factor=0.5,
            line_width=5
        )

        # draw easy axes
        end = origin + easy_axes[i] / np.linalg.norm(easy_axes[i])  # Normalized direction
        mlab.plot3d(
            [origin[0], end[0]],
            [origin[1], end[1]],
            [origin[2], end[2]],
            tube_radius=0.05,
            color=(0, 0, 1)
        )

    if filename is not None:
        if frame is None:
            mlab.savefig(f'{filename}.png')
            print(f'{filename}.png is saved!')
        else:
            mlab.savefig(f'{filename}_frame{frame}.png')
            mlab.clf()
            print(f'{filename}_frame{frame}.png is saved!')
    elif frame is not None:
        print("no filename to save the figure!")

    # Show the plot
    mlab.show()


def render_in_background(box_size, particles, *, filename=None):
    process = multiprocessing.Process(target=render_box_and_particles,
                                      args=(box_size, particles),
                                      kwargs={"filename": filename})
    process.start()
    return process


if __name__ == '__main__':
    dimension = np.array([200, 200, 200])
    n = 100
    part = (
        np.random.rand(n, 3) * dimension,
        np.random.rand(n, 3) - 0.5,
        np.random.rand(n, 3) - 0.5,
    )
    pros = render_in_background(
        dimension, part, filename='fff'
    )

    print('continue!')

    for i in range(100):
        print(i)

    pros.join()
    print('finished!')
