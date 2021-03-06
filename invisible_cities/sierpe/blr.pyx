import  numpy as np
cimport numpy as np
from scipy import signal as SGN

cpdef deconvolve_signal(double [:] signal_daq,
                        double coeff_clean            = 2.905447E-06,
                        double coeff_blr              = 1.632411E-03,
                        double thr_trigger            =     5,
                        int    accum_discharge_length =  5000):

    """
    The accumulator approach by Master VHB
    decorated and cythonized  by JJGC
    Current version using memory views

    In this version the recovered signal and the accumulator are
    always being charged. At the same time, the accumulator is being
    discharged when there is no signal. This avoids runoffs
    """

    cdef double coef = coeff_blr
    cdef double thr_acum = thr_trigger / coef
    cdef int len_signal_daq = len(signal_daq)

    cdef double [:] signal_r = np.zeros(len_signal_daq, dtype=np.double)
    cdef double [:] acum     = np.zeros(len_signal_daq, dtype=np.double)

    cdef int j

    # compute noise
    cdef double noise =  0
    cdef int nn = 400 # fixed at 10 mus

    for j in range(nn):
        noise += signal_daq[j] * signal_daq[j]
    noise /= nn
    cdef double noise_rms = np.sqrt(noise)

    # trigger line
    cdef double trigger_line = thr_trigger * noise_rms

    # cleaning signal
    cdef double [:]  b_cf
    cdef double [:]  a_cf

    b_cf, a_cf = SGN.butter(1, coeff_clean, 'high', analog=False);
    signal_daq = SGN.lfilter(b_cf, a_cf, signal_daq)

    cdef int k
    j = 0
    signal_r[0] = signal_daq[0]
    for k in range(1, len_signal_daq):

        # always update signal and accumulator
        signal_r[k] = (signal_daq[k] + signal_daq[k]*(coef / 2) +
                       coef * acum[k-1])

        acum[k] = acum[k-1] + signal_daq[k]

        if (signal_daq[k] < trigger_line) and (acum[k-1] < thr_acum):
            # discharge accumulator

            if acum[k-1] > 1:
                acum[k] = acum[k-1] * (1 - coef)
                if j < accum_discharge_length - 1:
                    j = j + 1
                else:
                    j = accum_discharge_length - 1
            else:
                acum[k] = 0
                j = 0
    # return recovered signal
    return np.asarray(signal_r)