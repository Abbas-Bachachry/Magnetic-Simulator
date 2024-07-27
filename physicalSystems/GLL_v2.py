import numpy as np
from numpy import pi
from numba import njit, prange


@njit(nogil=True, parallel=True)
def make_jacobian_3d(num_pos, pos, jacob):
    """
    make the jacobian matrix for particles in the box

    :param num_pos: number of the particles in the box
    :param pos: 1d array contain the position of the particles
                 [x1, y1, z1, x2, y2, z2, ..., xn, yn, zn,].
    :param jacob: a zeros array with shap [n, n] where n is the number of particles.
                    this is for putting data in
    :return: the jacob array
    """

    for posno in prange(num_pos):
        for i in range(num_pos):
            xi = pos[i * 3]
            yi = pos[i * 3 + 1]
            zi = pos[i * 3 + 2]
            xj = pos[posno * 3]
            yj = pos[posno * 3 + 1]
            zj = pos[posno * 3 + 2]
            xij = xi - xj
            yij = yi - yj
            zij = zi - zj
            rij = np.sqrt(xij ** 2 + yij ** 2 + zij ** 2)
            rij = np.round(rij, 10)
            if rij != 0:
                jacob[3 * i][posno * 3] = 3 * xij * xij / (4 * pi * rij ** 5) - 1 / (4 * pi * rij ** 3)  # x
                jacob[3 * i][posno * 3 + 1] = 3 * xij * yij / (4 * pi * rij ** 5)
                jacob[3 * i][posno * 3 + 2] = 3 * (xij * zij) / (4 * pi * rij ** 5)
                jacob[3 * i + 1][posno * 3] = 3 * xij * yij / (4 * pi * rij ** 5)  # Phi
                jacob[3 * i + 1][posno * 3 + 1] = 3 * yij * yij / (4 * pi * rij ** 5) - 1 / (4 * pi * rij ** 3)
                jacob[3 * i + 1][posno * 3 + 2] = 3 * (yij * zij) / (4 * pi * rij ** 5)
                jacob[3 * i + 2][posno * 3] = 3 * zij * xij / (4 * pi * rij ** 5)  # z
                jacob[3 * i + 2][posno * 3 + 1] = 3 * zij * yij / (4 * pi * rij ** 5)
                jacob[3 * i + 2][posno * 3 + 2] = 3 * (zij ** 2) / (4 * pi * rij ** 5) - 1 / (4 * pi * rij ** 3)

    return jacob * pi * 4e-7


@njit(nogil=True, parallel=True)
def prepare_llg_history(no, step, hs, positions, moment_unit_vectors, jacob, hk,
                        maxiterations, dampingfactor, easy_axis_unit_vectors, volumes, ms, moment_vectors_history):
    """

    :param no: number of FORCs
    :param step: step size
    :param hs: saturation filed
    :param positions: aray of position of the particle in the box
    :param moment_unit_vectors: moment unit vectors of the particles
    :param jacob: jacobian of interaction filed
    :param hk: intrinsic switching field of the particle (in Tesla)
    :param maxiterations: number of iteration
    :param dampingfactor: damping factor
    :param easy_axis_unit_vectors: direction of easy axis of particles
    :param volumes: volumes of particles
    :param ms: spontaneous magnetization of the particles
    :param moment_vectors_history: np.zeros([number of particles, number of curves, 3])
    :return: moment_vectors_history
    """

    stray_field = np.zeros(positions.shape)
    """interaction field among particles"""
    for j in np.arange(0, no):
        h = hs - j * step

        moment_vectors = np.zeros(moment_unit_vectors.size)
        """moment vectors in one row
            put the moment vectors in one row to calculate the stray filed.
            """
        moment_vectors[::3] = moment_unit_vectors[:, 0] * (volumes * ms).T
        moment_vectors[1::3] = moment_unit_vectors[:, 1] * (volumes * ms).T
        moment_vectors[2::3] = moment_unit_vectors[:, 2] * (volumes * ms).T
        num_pos = positions.shape[0]
        "number of positions (particles)"

        # Repeat the loop for a maximum number of 'max iterations'
        for _ in np.arange(maxiterations):
            # step 1 calculate stray filed
            stray_field[:, 0] = (jacob @ moment_vectors)[::3].T
            stray_field[:, 1] = (jacob @ moment_vectors)[1::3].T
            stray_field[:, 2] = (jacob @ moment_vectors)[2::3].T
            moment_unit_vectors, residuals = do_llg_start(num_pos, h, dampingfactor, moment_unit_vectors,
                                                          easy_axis_unit_vectors, stray_field, hk)
            residual_results = residuals / num_pos

            if residual_results < 1e-4:
                break
        moment_vectors_history[:, j, :] = moment_unit_vectors[:, :]
    return moment_vectors_history


@njit(nogil=True)
def do_llg_start(num_pos, h, f, moment_vector, ea_vector, stray_vector, hk):
    """

    :param num_pos: number of positions (particles)
    :param h: applied field
    :param f: damping factor
    :param moment_vector: moment vector before calculation
    :param ea_vector: direction of easy axis of particles
    :param stray_vector: interaction field among particles
    :param hk: intrinsic switching field of the particle (in Tesla)
    :return: new moment vector of particles, residuals
    """
    # Set applied field along z axis
    happ_vector = np.zeros(3)
    happ_vector[2] = h
    residuals = 0

    for k in prange(num_pos):  # (k=0;k < no_particles;k += 1)
        # Calculate component of M along easy axis
        mz = moment_vector[k] @ ea_vector[k]

        # Effective field for uniaxial anisotropy
        heff_vector = happ_vector + stray_vector[k] + hk[k] * mz * ea_vector[k]  # last one is the anisotropy filed

        mag = (heff_vector[0] ** 2 + heff_vector[1] ** 2 + heff_vector[2] ** 2) ** 0.5
        heff_vector /= mag

        # Calculate residual
        torque_vector = np.cross(moment_vector[k], heff_vector)

        # Set global residual wave to magnitude of residual vector
        residuals += (torque_vector[0] ** 2 + torque_vector[1] ** 2 + torque_vector[2] ** 2) ** 0.5

        # Calculate change in moment direction
        """Specify fraction f of required change to be applied. 
        Places new vector not exactly along calculated Heff, but a fraction of the way towards it."""
        moment_vector[k] = moment_vector[k] * (1 - f) + f * heff_vector
        "This method is equivalent to minimizing the free energy of the system"

        # Normalise new moment vector and Heff
        mag = (moment_vector[k][0] ** 2 + moment_vector[k][1] ** 2 + moment_vector[k][2] ** 2) ** 0.5
        moment_vector[k] /= mag

    return moment_vector, residuals


@njit(nogil=True, parallel=True)
def llg_forc(no, hs, step, moment_vectors_history, jacobian, easy_axis_unit_vectors, hk,
             volumes, ms, matrix2, max_iteration, dumping_factor):
    """ this function calculate the simulated FORC using llg method

    note that the strayfield, the moment_and_residual, and matrix2 added only because numba.jit cannot create
    2-dimensional array inside function in parallel mode. the matrix2 is calculated here.

    the strayfield array is the array that is going to contain the stray of filed applied to each particle.
    the moment_and_residual is going to contain the moment and the residual of each particle.
    the metrix2 is going to contain the simulated FORC data.

    the moment_vectors_history contain the moment vector of every particle at every reversal filed. this is needed as an
    initial state to calculate the magnetization for other part of the curves.

    :param int no: number of curves
    :param float|int hs: saturation field
    :param float step: field interval
    :param np.ndarray moment_vectors_history: an array of floats with shape of (number of particles, number of curves, 3) containing the history of moment vectors
    :param np.ndarray jacobian: an array with shape of (3*number of particles, 3*number of particles) containing jacobian
    :param np.ndarray easy_axis_unit_vectors: an array with the shape of (number of particles, 3) containing the easy axes  of particles
    :param np.ndarray hk: an array with the shape of (number of particles, 3) containing the coercivity of particles
    :param float volumes: volume of each particle (supposing they have the same volume)
    :param np.ndarray ms: saturation magnetization
    :param np.ndarray matrix2: here the stray field array is np.zeros([number of curves, number of curves]); added only because numba can't creat 2dimensional array within parallel mode
    :param int max_iteration: maximum nuber of iteration
    :param float dumping_factor: a number between 0 and 1
    :return: the measured curves
    """
    for i in prange(no):
        ha = hs - i * step
        matrix2[i] = llg_curve(no, i, ha, step, moment_vectors_history, jacobian,
                               easy_axis_unit_vectors, hk, volumes,
                               ms, max_iteration, dumping_factor)
    return matrix2


@njit(nogil=True)
def llg_curve(no, i, ha, step, momhis, jacob, easy, hk, vol, ms, maxiter, f):
    """this function calculate the ith curve using llg method.

    note that the strayfield and the moment_and_residual added only because numba.jit cannot create
    2-dimensional array inside function in parallel mode. the stray field is calculated here.

    the moment_vectors_history contain the moment vector of every particle at every reversal filed. this is needed as an
    initial state to calculate the magnetization for other part of the curves.

    :param int no: number of curves
    :param int i: the ith Forc index (i)
    :param float ha: reversal filed
    :param float step: field interval
    :param np.ndarray momhis: an array of floats with shape of (number of particles, number of curves, 3) containing the history of moment vectors
    :param np.ndarray jacob: an array with shape of (3*number of particles, 3*number of particles) containing jacobian
    :param np.ndarray easy: an array with the shape of (number of particles, 3) containing the easy axes  of particles
    :param np.ndarray hk: an array with the shape of (number of particles, 3) containing the coercivity of particles
    :param float vol: volume of each particle (supposing they have the same volume)
    :param np.ndarray ms: saturation magnetization
    :param int maxiter: maximum nuber of iteration
    :param float f: a number between 0 and 1 (0.9 is preferable)
    :return: the calculated ith curve
    """

    moment_unit_vectors = momhis[:, i, :]
    num_particles = moment_unit_vectors.shape[0]
    residual = np.zeros(num_particles)
    m2 = np.zeros(no)
    m2[:] = np.nan
    m = np.empty(3 * num_particles)
    for j in range(i + 1):
        hb = ha + j * step
        for l in range(maxiter):
            m[0::3] = moment_unit_vectors[:, 0] * vol * ms
            m[1::3] = moment_unit_vectors[:, 1] * vol * ms
            m[2::3] = moment_unit_vectors[:, 2] * vol * ms
            moment_reference_wave = llg_mt(hb, moment_unit_vectors, (jacob @ m), easy, hk, f)
            for k in range(num_particles):
                w = moment_reference_wave[4 * k:4 * (k + 1)]
                if not np.isnan(w[0]):
                    moment_unit_vectors[k, :3] = w[:3]
                    residual[k] = w[3]

            if np.mean(residual) < 1e-4:
                break

        moment = moment_unit_vectors[:, 2]
        m2[j + no - i - 1] = np.mean(moment)
    return m2


@njit(nogil=True)
def llg_mt(h, moment_vector, stray_field, ea_vector, hk, f):
    """this function calculate the moment of the system at applied filed h with previous state of moment_vector

    note that the moment_and_residual added only because numba.jit cannot create
    2-dimensional array inside function in parallel mode. the moment_and_residual is calculated here.

    :param float h: applied filed
    :param moment_vector: an array with the shape of (number of particles, 3) that contain the moment unit vector of each particle at previous state
    :param stray_field: an array containing the stray filed with shape of (number of particles, 3)
    :param ea_vector: an array with the shape of (number of particles, 3) containing the easy axes  of particles
    :param hk: an array with the shape of (number of particles, 3) containing the coercivity of particles
    :param f: dumping factor
    :return: the moment and residual of the next state
    """
    applied_field = np.zeros(3)
    applied_field[2] = h  # applied filed in the Z direction
    moment_and_residual = np.zeros(4 * moment_vector.shape[0])
    # Effective field for uniaxial anisotropy
    for k in prange(moment_vector.shape[0]):
        mz = moment_vector[k] @ ea_vector[k].T
        effective_filed = applied_field + stray_field[3 * k:3 * (k + 1)] + hk[k] * mz * ea_vector[k]
        # Normalise effective field. This serves as new value for the moment vector
        mag = (effective_filed[0] ** 2 + effective_filed[1] ** 2 + effective_filed[2] ** 2) ** 0.5
        effective_filed /= mag
        # Calculate residual using normalised effecting field
        torque_vector = np.cross(moment_vector[k], effective_filed)

        # Set global residual wave to magnitude of residual vector
        moment_and_residual[4 * k + 3] = (torque_vector[0] ** 2 + torque_vector[1] ** 2 + torque_vector[2] ** 2) ** 0.5
        # Calculate change in moment direction
        # Specify fraction f of required change to be applied. Places new vector not exactly along calculated Heff,
        # but a fraction of the way towards it.
        moment_vector[k] = moment_vector[k] * (1 - f) + f * effective_filed
        # Normalise new moment vector
        mag = (moment_vector[k, 0] ** 2 + moment_vector[k, 1] ** 2 + moment_vector[k, 2] ** 2) ** 0.5
        moment_vector[k] /= mag
        moment_and_residual[4 * k:4 * k + 3] = moment_vector[k]
        # Set 1st three points of result wave to new result
    return moment_and_residual
