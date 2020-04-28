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


cpdef deconv_pmt(np.ndarray[double, ndim=2] pmtrwf,
                 double [:]                 coeff_c,
                 double [:]                 coeff_blr,
                 list                       pmt_active             =    [],
                 double                     thr_trigger            =     5,
                 int                        accum_discharge_length =  5000):
    """
    Deconvolve all the PMTs in the event.
    :param pmtrwf: array of PMTs holding the pedestal subtracted waveform
    :param coeff_c:     cleaning coefficient
    :param coeff_blr:   deconvolution coefficient
    :param pmt_active:  list of active PMTs (by id number). An empt list
                        implies that all PMTs are active
    :param thr_trigger: threshold to start the BLR process
    
    :returns: an array with deconvoluted PMTs. If PMT is not active
              wvfs are removed.
    """

    cdef int NPMT = pmtrwf.shape[0]
    cdef int NWF  = pmtrwf.shape[1]
    cdef double [:, :] signal_i = pmtrwf.astype(np.double)
    cdef double [:]    signal_r = np.zeros(NWF, dtype=np.double)
    CWF = []

    cdef list PMT = list(range(NPMT))
    if len(pmt_active) > 0:
        PMT = pmt_active

    cdef int pmt
    for pmt in PMT:
        signal_r = deconvolve_signal(signal_i[pmt],
                                     coeff_clean            = coeff_c[pmt],
                                     coeff_blr              = coeff_blr[pmt],
                                     thr_trigger            = thr_trigger,
                                     accum_discharge_length = accum_discharge_length)

        CWF.append(signal_r)

    return np.array(CWF)
