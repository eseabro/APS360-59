import numpy as np
import pandas as pd
import scipy.signal
import csv


class feature_extractor():
    def __init__(self, dataf, name):
        self.df = (dataf-dataf.mean())/dataf.std()
        self.fname = name
        self.N = self.df.shape[0]
        self.sampling_rate = 2*60*60
        self.head_cols = ["Average", "Standard Deviation", "Kurtosis", "Variance", "Peak Frequency", 
                     "Min Freq", "Frequency Centroid", "Amplitude", "Skewness", 
                     "Zero Crossing", "Zero Count"]
        
        self.header = np.array([])
        for col in self.df.columns:
            out_a = np.array([col+' ' + x for x in self.head_cols])
            self.header = np.concatenate((self.header, out_a))


        # Time series properties
        self.avg = self.get_avg()
        self.std = self.get_std()
        self.kurt = self.get_kurt()
        self.var = self.get_var()

        # Frequency Domain Properties
        self.fft_amp = self.df.apply(np.fft.fft)
        self.fft_freqs = np.fft.fftfreq(self.N)
        self.rfft = self.df.apply(np.real)
        self.fundamental_freq = 1/(self.sampling_rate*self.N)

        self.amplitude = []
        self.min_freq = []
        self.peak_freq = []
        self.get_fft_props()

        self.freq_cen = self.get_freq_cen()
        self.skewness = self.get_skew()
        self.zero_c, self.zero_n = self.get_zero()

        self.out_features = np.array([self.avg, self.std, self.kurt, self.var, self.peak_freq, 
                     self.min_freq, self.freq_cen, self.amplitude, self.skewness, 
                     self.zero_c, self.zero_n]).flatten()

        self.save_features()


    def get_avg(self):
        return self.df.mean(skipna=True).to_list()
    
    def get_std(self):
        return self.df.std(skipna=True).to_list()
    
    def get_kurt(self):
        return self.df.kurtosis(skipna=True).to_list()
    
    def get_var(self):
        return self.df.var(skipna=True).to_list()
    
    def get_fft_props(self):
        for col in self.fft_amp.columns:
            idx = np.argmax(np.abs(self.fft_amp[col]))
            min_idx = np.argmin(np.abs(self.fft_amp[col]))

            freq = self.fft_freqs[idx]
            freq_in_hertz = abs(freq * self.sampling_rate)
            self.peak_freq.append(freq_in_hertz)
            self.amplitude.append(np.max(np.abs(self.fft_amp[col])))
            self.min_freq.append(abs(self.fft_freqs[min_idx] * self.sampling_rate))
        return True
    
    def get_freq_cen(self):
        f_out = []
        for col in self.rfft.columns:
            f_out.append(np.sum(self.rfft[col]*self.fft_freqs) / np.sum(self.rfft[col]))
        return f_out

    def get_skew(self):
        skout = []
        for col in self.rfft:
            skout.append(scipy.stats.skew(self.rfft[col]))
        return skout
    
    def get_zero(self):
        z_out = []
        z_count = []
        for col in self.df:
            zero_crossings = np.where(np.diff(np.sign(self.df[col])))[0]
            z_out.append(zero_crossings[0])
            z_count.append(len(zero_crossings))
        return z_out, z_count
    

    def save_features(self):
        with open(self.fname, 'w+', newline='') as f:
            wr = csv.writer(f, delimiter=',')
            wr.writerow(self.header)
            wr.writerow(self.out_features)
            f.close()
        
        print("Written")
        return
    