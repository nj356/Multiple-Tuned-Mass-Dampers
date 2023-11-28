#! /usr/bin/env python3

import argparse
import numpy as np
import scipy.integrate
import matplotlib.pyplot as plt


def MLKF_1dof(m1, l1, k1, f1):

    """Return mass, damping, stiffness & force matrices for 1DOF system"""

    M = np.array([[m1]])
    L = np.array([[l1]])
    K = np.array([[k1]])
    F = np.array([f1])

    return M, L, K, F


def MLKF_2dof(m1, l1, k1, f1, m2, l2, k2, f2):

    """Return mass, damping, stiffness & force matrices for 2DOF system"""

    M = np.array([[m1, 0], [0, m2]])
    L = np.array([[l1+l2, -l2], [-l2, l2]])
    K = np.array([[k1+k2, -k2], [-k2, k2]])
    F = np.array([f1, f2])

    return M, L, K, F


def freq_response(w_list, M, L, K, F):

    """Return complex frequency response of system"""

    return np.array(
        [np.linalg.solve(-w*w * M + 1j * w * L + K, F) for w in w_list]
    )


def time_response(t_list, M, L, K, F):

    """Return time response of system"""

    mm = M.diagonal()

    def slope(t, y):
        xv = y.reshape((2, -1))
        a = (F - L@xv[1] - K@xv[0]) / mm
        s = np.concatenate((xv[1], a))
        return s

    solution = scipy.integrate.solve_ivp(
        fun=slope,
        t_span=(t_list[0], t_list[-1]),
        y0=np.zeros(len(mm) * 2),
        method='Radau',
        t_eval=t_list
    )

    return solution.y[0:len(mm), :].T


def last_nonzero(arr, axis, invalid_val=-1):

    """Return index of last non-zero element of an array"""

    mask = (arr != 0)
    val = arr.shape[axis] - np.flip(mask, axis=axis).argmax(axis=axis) - 1
    return np.where(mask.any(axis=axis), val, invalid_val)


def plot(fig, hz, sec, M, L, K, F, show_phase=None):

    """Plot frequency and time domain responses"""

    # Generate response data

    f_response = freq_response(hz * 2*np.pi, M, L, K, F)
    f_amplitude = np.abs(f_response)
    t_response = time_response(sec, M, L, K, F)

    # Determine suitable legends

    f_legends = (
        'm{} peak {:.4g} metre at {:.4g} Hz'.format(
            i+1,
            f_amplitude[m][i],
            hz[m]
        )
        for i, m in enumerate(np.argmax(f_amplitude, axis=0))
    )

    equilib = np.abs(freq_response([0], M, L, K, F))[0]         # Zero Hz
    toobig = abs(100 * (t_response - equilib) / equilib) >= 2
    lastbig = last_nonzero(toobig, axis=0, invalid_val=len(sec)-1)

    t_legends = (
        'm{} settled to 2% beyond {:.4g} sec'.format(
            i+1,
            sec[lastbig[i]]
        )
        for i, _ in enumerate(t_response.T)
    )

    # Create plot

    fig.clear()

    if show_phase is not None:
        ax = [
            fig.add_subplot(3, 1, 1),
            fig.add_subplot(3, 1, 2),
            fig.add_subplot(3, 1, 3)
        ]
        ax[1].sharex(ax[0])
    else:
        ax = [
            fig.add_subplot(2, 1, 1),
            fig.add_subplot(2, 1, 2)
        ]

    ax[0].set_title('Amplitude of frequency domain response to sinusoidal force')
    ax[0].set_xlabel('Frequency/hertz')
    ax[0].set_ylabel('Amplitude/metre')
    ax[0].legend(ax[0].plot(hz, f_amplitude), f_legends)

    if show_phase is not None:
        p_legends = (f'm{i+1}' for i in range(f_response.shape[1]))

        f_phases = f_response
        if show_phase == 0:
            ax[1].set_title(f'Phase of frequency domain response to sinusoidal force')
        else:
            f_phases /= f_response[:, show_phase-1:show_phase]
            ax[1].set_title(f'Phase, relative to m{show_phase}, of frequency domain response to sinusoidal force')
        f_phases = np.degrees(np.angle(f_phases))

        ax[1].set_xlabel('Frequency/hertz')
        ax[1].set_ylabel('Phase/°')
        ax[1].legend(ax[1].plot(hz, f_phases), p_legends)

    ax[-1].set_title('Time domain response to step force')
    ax[-1].set_xlabel('Time/second')
    ax[-1].set_ylabel('Displacement/metre')
    ax[-1].legend(ax[-1].plot(sec, t_response), t_legends)

    fig.tight_layout()


def arg_parser():
    ap = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='''
            For a system with one or two degrees of freedom, show the
            frequency domain response to an applied sinusoidal force,
            and the time domain response to an step force.
    ''')

    ap.add_argument('--m1', type=float, default=7.88, help='Mass 1')
    ap.add_argument('--l1', type=float, default=3.96, help='Damping 1')
    ap.add_argument('--k1', type=float, default=4200, help='Spring 1')
    ap.add_argument('--f1', type=float, default=0.25, help='Force 1')

    ap.add_argument('--m2', type=float, default=None, help='Mass 2')
    ap.add_argument('--l2', type=float, default=1, help='Damping 2')
    ap.add_argument('--k2', type=float, default=106.8, help='Spring 2')
    ap.add_argument('--f2', type=float, default=0, help='Force 2')

    ap.add_argument(
        '--hz', type=float, nargs=2, default=(0, 5),
        help='Frequency range'
    )
    ap.add_argument(
        '--sec', type=float, default=30,
        help='Time limit'
    )

    ap.add_argument(
        '--show-phase', type=int, nargs='?', const=0,
        help='''Show the frequency domain phase response(s).
        If this option is given without a value then phases are shown
        relative to the excitation.
        If a value is given then phases are shown relative to the
        phase of the mass with that number.
    ''')

    return ap


def main():

    """Main program"""

    # Read command line

    ap = arg_parser()
    args = ap.parse_args()

    # Generate matrices describing the system

    if args.m2 is None:
        M, L, K, F = MLKF_1dof(
            args.m1, args.l1, args.k1, args.f1
        )
    else:
        M, L, K, F = MLKF_2dof(
            args.m1, args.l1, args.k1, args.f1,
            args.m2, args.l2, args.k2, args.f2
        )

    # Generate frequency and time arrays

    hz = np.linspace(args.hz[0], args.hz[1], 10001)
    sec = np.linspace(0, args.sec, 10001)

    # Plot results

    fig = plt.figure()
    plot(fig, hz, sec, M, L, K, F, args.show_phase)
    fig.canvas.mpl_connect('resize_event', lambda x: fig.tight_layout(pad=2.5))
    plt.show()


if __name__ == '__main__':
    main()
