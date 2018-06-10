import time
import pandas as pd
import cv2
import numpy as np
from numpy import rad2deg
from scipy.signal import argrelextrema, savgol_filter
import scipy
from itertools import product
from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh


class walk_analysis:


    def __init__(self, path, framerate = 24, fetched = False, path_csv = None):
        """
        Explanation of the results
        There are 2 main categories:

        1.degreeprofile:
            this profile measure the range of motion and the consitency between indivual walking cycles
        2.distanceprofile:
            this profile measures the stepdistance and the distance between fot and hip as a validator for the food distances

        Within these 2 categories there are 2 main characteristics:
            1. Movement pace: Indicates how fast the individual cycles were completed and consistent the indicators in the group were
            2. Movement consistency: Indicates how regular the single cycles of the movements were

        """
        self.fetched = fetched
        self.path_csv = path_csv
        self.path = path
        self.dataframe = None
        self.augmented_frame = None
        self.degree_pairs = {"r_fk": ["09", "10", "08","09"],
                                "l_fk": ["12", "13","11", "12"],
                                    "r_hk": ["08", "01", "08", "09"],
                                        "l_hk" : ["01", "11", "11", "12"]}

        self.distance_pairs = {"r_d": ["08", "10"], "l_d": ["11","13"], "g_d": ["10", "13"]}
        self.framerate = framerate
        self.raw_walk_data = None
        self.idealised_walk_data = None


    def create_dataframe(self, path, model='cmu', resize_out_ratio=4.0, resize = "432x368"):
        """Analyzes the frame and returns a jupyter array where the relativ positons of the joints are located in the picture"""
        a = [str(x) if x > 9 else '0' + str(x) for x in range(1,19)]
        b = ['x', 'y', 'score']
        temp_dic = {'bp' + str(key[0]) + str(key[1]) : [] for key in product(a,b) }
        frame = 0
        logger.debug('initialization %s : %s' % (model, get_graph_path(model)))
        w, h = model_wh(resize)
        e = TfPoseEstimator(get_graph_path(model), target_size=(w, h))
        cap = cv2.VideoCapture(path)

        while cap.isOpened():
            frame += 1
            if frame % 100 == 0:
                print(frame)

            ret_val, image = cap.read()
            try:
                humans = e.inference(image, resize_to_default=(w > 0 and h > 0), upsample_size=resize_out_ratio)
            except:
                break

            for human in humans:
                for key, value in temp_dic.items():
                    key = int(key[2:4])
                    found = human.body_parts.keys()
                    if key in found:
                        #print("found something")
                        skey = str(key) if key > 9 else '0'+ str(key)
                        temp_dic['bp' + skey+"score"].append(human.body_parts[key].score)
                        temp_dic['bp' + skey+"y"].append(human.body_parts[key].x)
                        temp_dic['bp' + skey+"x"].append(human.body_parts[key].y)
                    else:
                        skey = str(key) if key > 9 else '0'+str(key)
                        temp_dic['bp' + skey+"score"].append(None)
                        temp_dic['bp' + skey+"y"].append(None)
                        temp_dic['bp' + skey+"x"].append(None)

        cv2.destroyAllWindows()

        self.df = pd.DataFrame(temp_dic)
        


    def augment_data(self):
        """converts the relative position of the joins to vectors and finnaly to angles"""


        def unit_vector(vector):
            return vector / np.linalg.norm(vector)


        def angle_between(v1, v2):
            v1_u = unit_vector(v1)
            v2_u = unit_vector(v2)
            return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


        def calc_angle(x_1, y_1, x_2, y_2, x_3, y_3, x_4, y_4):
            v1 = np.array([abs(x_1-x_2), abs(y_1-y_2)])
            v2 = np.array([abs(x_3-x_4),abs(y_3-y_4)])
            v1 = unit_vector(v1)
            v2 = unit_vector(v2)
            radian = angle_between(v1, v2)
            return rad2deg(radian)


        def calc_distance(x_1, y_1, x_2, y_2):
            a = abs(x_1-x_2)
            b = abs(y_1-y_2)
            return np.sqrt(a**2 + b**2)


        def apply_degrees(odf, pairs):
            df = odf.copy(deep=True)
            for key, value in pairs.items():
                k1, k2, k3, k4 = "bp" + value[0] + "x", "bp" + value[0] + 'y', "bp" +  value[1] + "x","bp" + value[1] + 'y'
                k5, k6, k7, k8 = "bp" + value[2] + "x", "bp" + value[2] + 'y', "bp" +  value[3] + "x","bp" + value[3] + 'y'
                df[key] = df.apply(lambda x: calc_angle(x[k1], x[k2], x[k3], x[k4], x[k5], x[k6], x[k7], x[k8]), axis = 1)
            return df


        def apply_distance(odf, pairs):
            df = odf.copy(deep=True)
            for key, value in pairs.items():
                k1, k2, k3, k4 = "bp" + value[0] + "x", "bp" + value[0] + 'y', "bp" +  value[1] + "x","bp" + value[1] + 'y'
                df[key] = df.apply(lambda x: calc_distance(x[k1], x[k2], x[k3], x[k4]), axis = 1)
            return df
        

        self.augmented_frame = apply_degrees(self.dataframe, self.degree_pairs) 
        self.augmented_frame = apply_distance(self.augmented_frame, self.distance_pairs)



    def calculate_cycles(self, series, set_global=False):
        """return the quality for a single hyperparameter"""
    
        def fit_polynom(series):
            """Fits a low order polynomial function to pseudo periodic input data"""
            #optional implementation
        

        def fit_sin(series):
            """Fits a sine function to periodic input data"""
            tt = series.index.values
            yy = series.values
            ff = np.fft.fftfreq(len(tt), (tt[1]-tt[0]))   
            Fyy = abs(np.fft.fft(yy))
            guess_freq = abs(ff[np.argmax(Fyy[1:])+1])
            guess_amp = np.std(yy) * 2.**0.5
            guess_offset = np.mean(yy)
            guess = np.array([guess_amp, 2.*np.pi*guess_freq, 0., guess_offset])
            def sinfunc(t, A, w, p, c):  return A * np.sin(w*t + p) + c
            popt, pcov = scipy.optimize.curve_fit(sinfunc, tt, yy, p0=guess)
            A, w, p, c = popt
            f = w/(2.*np.pi)
            fitfunc = lambda t: A * np.sin(w*t + p) + c
            return fitfunc, popt


        def normalize_series(series, fitfunc):
            """normalizes input data"""
            tt = series.index.values
            yy = series.values
            yy = [np.log(y - fitfunc(x)) if y >= fitfunc(x) else -np.log(abs(y - fitfunc(x)))  for x,y in zip(tt,yy)]
            return pd.Series(yy)
        
        
        def calc_period_hyperparameters(series, set_global = False):
            """Calculates the period and the consitency of the angle profile"""
            
            
            def calc_period(omega):
                return 2*np.pi/omega
            
            
            def calc_stability(y_act, y_beta):
                y_act = argrelextrema(y_act, np.greater)[0]
                y_beta = argrelextrema(y_beta, np.greater)[0]
                displacements = [min([abs(a-x) for a in y_act ]) for x in y_beta]
                return displacements
            
            
            series = series.fillna(method='ffill').fillna(0)
            x = self.augmented_frame.index.values
            alpha, alpha_props = fit_sin(series)
            y = normalize_series(series, alpha)
            beta, beta_props = fit_sin(y)

            y_beta = np.array([2 * beta(a) for a in x])
            y_ = savgol_filter(y, 21,3)
            if set_global:
                self.raw_walk_data = y_
                self.idealised_walk_data = y_beta
            
            dis = calc_stability(y.values, np.array(y_beta))
            per = calc_period(beta_props[1])
    #         plt.plot(x,y)
    #         plt.plot(x, y_beta)
    #         plt.show()
    #         plt.hist(dis)
            return {"Stability": dis, "period": per}
        
        return calc_period_hyperparameters(series, set_global)

    
    def aggregate_cycles(self):
        """Analysis of all the metric combined"""
        #distance cycle analysis
        dp = []
        omegas = []

        for key in self.distance_pairs:
            t_res = self.calculate_cycles(self.augmented_frame[key])
            dp.append(t_res['Stability'])
            omegas.append(t_res['period'])

        omegas = [x/self.framerate for x in omegas]
        omega_mean = np.mean(omegas)
        omega_std = np.std(omegas)
        dp = [np.mean(x) for x in dp]
        dp_mean = np.mean(dp)
        dp_std = np.std(dp)
        self.distance_profile = {"Movement_pace": {"mean": omega_mean, "std": omega_std},
                                "Movement_consistency": {"mean": dp_mean, "std": dp_std}}
        
        #angle cycle analysis
        dp = []
        omegas = []

        for key in self.degree_pairs:
            set_glob = False
            if key == 'r_hk':
                set_glob = True
            t_res = self.calculate_cycles(self.augmented_frame[key], set_glob)
            dp.append(t_res['Stability'])
            omegas.append(t_res['period'])

        omegas = [x/self.framerate for x in omegas]
        omega_mean = np.mean(omegas)
        omega_std = np.std(omegas)
        dp = [np.mean(x) for x in dp]
        dp_mean = np.mean(dp)
        dp_std = np.std(dp)
        self.degree_profile = {"Movement_pace": {"mean": omega_mean, "std": omega_std},
                                "Movement_consistency": {"mean": dp_mean, "std": dp_std}}



    def create_walking_profile(self):
        """Calculates the target metrics by calling the helper functions"""
        if not self.fetched:
            self.create_dataframe()
        else:
            self.dataframe = pd.read_csv(self.path_csv)
        self.augment_data()
        self.aggregate_cycles()
        return {"profiles": {"degree_profile": self.degree_profile, "distance_profile": self.distance_profile}, "walk_data": {"raw_data": self.raw_walk_data.tolist(), "idealised_walk": self.idealised_walk_data.tolist()}}





