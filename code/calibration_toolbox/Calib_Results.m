% Intrinsic and Extrinsic Camera Parameters
%
% This script file can be directly executed under Matlab to recover the camera intrinsic and extrinsic parameters.
% IMPORTANT: This file contains neither the structure of the calibration objects nor the image coordinates of the calibration points.
%            All those complementary variables are saved in the complete matlab data file Calib_Results.mat.
% For more information regarding the calibration model visit http://www.vision.caltech.edu/bouguetj/calib_doc/


%-- Focal length:
fc = [ 1870.000295077482178 ; 1876.486326981806769 ];

%-- Principal point:
cc = [ 930.341426739959047 ; 548.745084322920320 ];

%-- Skew coefficient:
alpha_c = 0.000000000000000;

%-- Distortion coefficients:
kc = [ 0.170053259242693 ; -0.484440452847381 ; -0.007290351161580 ; -0.006406093836988 ; 0.000000000000000 ];

%-- Focal length uncertainty:
fc_error = [ 12.562276942649840 ; 10.444694031481722 ];

%-- Principal point uncertainty:
cc_error = [ 15.623612872729067 ; 25.243477539362839 ];

%-- Skew coefficient uncertainty:
alpha_c_error = 0.000000000000000;

%-- Distortion coefficients uncertainty:
kc_error = [ 0.023815039720652 ; 0.105496454696418 ; 0.002983764395177 ; 0.003380474183933 ; 0.000000000000000 ];

%-- Image size:
nx = 1920;
ny = 1080;


%-- Various other variables (may be ignored if you do not use the Matlab Calibration Toolbox):
%-- Those variables are used to control which intrinsic parameters should be optimized

n_ima = 36;						% Number of calibration images
est_fc = [ 1 ; 1 ];					% Estimation indicator of the two focal variables
est_aspect_ratio = 1;				% Estimation indicator of the aspect ratio fc(2)/fc(1)
center_optim = 1;					% Estimation indicator of the principal point
est_alpha = 0;						% Estimation indicator of the skew coefficient
est_dist = [ 1 ; 1 ; 1 ; 1 ; 0 ];	% Estimation indicator of the distortion coefficients


%-- Extrinsic parameters:
%-- The rotation (omc_kk) and the translation (Tc_kk) vectors for every calibration image and their uncertainties

%-- Image #1:
omc_1 = [ 1.956449e+00 ; -1.571573e-01 ; 1.522836e-01 ];
Tc_1  = [ -8.361937e+01 ; 3.816310e+01 ; 1.861283e+02 ];
omc_error_1 = [ 1.232466e-02 ; 5.807948e-03 ; 8.270875e-03 ];
Tc_error_1  = [ 1.591207e+00 ; 2.627937e+00 ; 1.658660e+00 ];

%-- Image #2:
omc_2 = [ 1.557647e+00 ; -2.597431e+00 ; 4.518488e-01 ];
Tc_2  = [ 2.463283e+02 ; -2.699625e+01 ; 7.757076e+02 ];
omc_error_2 = [ 7.530209e-03 ; 1.490183e-02 ; 2.286832e-02 ];
Tc_error_2  = [ 6.502373e+00 ; 1.041778e+01 ; 6.134270e+00 ];

%-- Image #3:
omc_3 = [ 2.167088e+00 ; -1.376219e+00 ; 5.682772e-01 ];
Tc_3  = [ 1.170477e+02 ; 1.055390e+02 ; 1.010463e+03 ];
omc_error_3 = [ 1.309571e-02 ; 1.184402e-02 ; 1.996169e-02 ];
Tc_error_3  = [ 8.505575e+00 ; 1.392588e+01 ; 9.217067e+00 ];

%-- Image #4:
omc_4 = [ 2.481008e+00 ; -8.643044e-01 ; 1.780733e-01 ];
Tc_4  = [ 6.536999e-01 ; 1.193833e+02 ; 5.680614e+02 ];
omc_error_4 = [ 1.189573e-02 ; 6.355404e-03 ; 1.584230e-02 ];
Tc_error_4  = [ 4.806576e+00 ; 8.001451e+00 ; 4.526430e+00 ];

%-- Image #5:
omc_5 = [ 8.647982e-01 ; 2.861048e+00 ; -6.588433e-01 ];
Tc_5  = [ 5.394297e+01 ; -2.157438e+02 ; 9.475180e+02 ];
omc_error_5 = [ 6.776278e-03 ; 1.492103e-02 ; 2.210266e-02 ];
Tc_error_5  = [ 7.982362e+00 ; 1.214462e+01 ; 7.340740e+00 ];

%-- Image #6:
omc_6 = [ 9.574449e-01 ; 2.631872e+00 ; -9.595912e-01 ];
Tc_6  = [ 6.851491e-01 ; -1.378572e+02 ; 6.464854e+02 ];
omc_error_6 = [ 5.088666e-03 ; 1.144585e-02 ; 1.577358e-02 ];
Tc_error_6  = [ 5.405423e+00 ; 8.278693e+00 ; 4.780563e+00 ];

%-- Image #7:
omc_7 = [ 2.503159e+00 ; 9.443415e-01 ; -2.088345e-01 ];
Tc_7  = [ -1.449875e+02 ; 4.255440e+00 ; 5.130821e+02 ];
omc_error_7 = [ 1.061605e-02 ; 5.852043e-03 ; 1.426746e-02 ];
Tc_error_7  = [ 4.300990e+00 ; 6.879824e+00 ; 3.915553e+00 ];

%-- Image #8:
omc_8 = [ 2.166683e+00 ; -1.766241e-01 ; 2.310184e-01 ];
Tc_8  = [ -7.636043e+01 ; 5.486303e+01 ; 2.362398e+02 ];
omc_error_8 = [ 1.169170e-02 ; 5.426310e-03 ; 9.411908e-03 ];
Tc_error_8  = [ 2.009587e+00 ; 3.333386e+00 ; 1.850250e+00 ];

%-- Image #9:
omc_9 = [ 4.552128e-02 ; 2.369223e+00 ; -1.987212e+00 ];
Tc_9  = [ 7.513483e+01 ; -6.668312e+00 ; 4.462613e+02 ];
omc_error_9 = [ 8.408355e-03 ; 1.394147e-02 ; 1.589548e-02 ];
Tc_error_9  = [ 3.708642e+00 ; 5.990771e+00 ; 2.253526e+00 ];

%-- Image #10:
omc_10 = [ 7.152724e-03 ; 2.601943e+00 ; -1.755298e+00 ];
Tc_10  = [ 9.582667e+01 ; -6.323973e+01 ; 4.452350e+02 ];
omc_error_10 = [ 7.893967e-03 ; 1.249247e-02 ; 1.672836e-02 ];
Tc_error_10  = [ 3.709488e+00 ; 5.849425e+00 ; 2.747305e+00 ];

%-- Image #11:
omc_11 = [ 8.066923e-01 ; -2.677793e+00 ; 8.100076e-01 ];
Tc_11  = [ 1.664717e+02 ; -1.018697e+01 ; 7.147322e+02 ];
omc_error_11 = [ 5.804617e-03 ; 1.325178e-02 ; 1.870516e-02 ];
Tc_error_11  = [ 5.976984e+00 ; 9.609475e+00 ; 5.636426e+00 ];

%-- Image #12:
omc_12 = [ 1.863936e+00 ; -2.033990e+00 ; 4.332503e-01 ];
Tc_12  = [ 2.599755e+02 ; 8.841677e+01 ; 8.828085e+02 ];
omc_error_12 = [ 1.147847e-02 ; 1.503874e-02 ; 2.368213e-02 ];
Tc_error_12  = [ 7.477294e+00 ; 1.219401e+01 ; 7.620226e+00 ];

%-- Image #13:
omc_13 = [ 1.497017e+00 ; -1.487740e+00 ; 9.994778e-01 ];
Tc_13  = [ 1.207716e+02 ; -3.282930e+01 ; 4.028894e+02 ];
omc_error_13 = [ 9.485018e-03 ; 1.129898e-02 ; 1.091840e-02 ];
Tc_error_13  = [ 3.363781e+00 ; 5.365750e+00 ; 3.100318e+00 ];

%-- Image #14:
omc_14 = [ 2.087365e+00 ; -2.069700e+00 ; 2.444334e-01 ];
Tc_14  = [ 1.355313e+02 ; 7.861213e+01 ; 3.517329e+02 ];
omc_error_14 = [ 6.430061e-03 ; 8.217784e-03 ; 1.468124e-02 ];
Tc_error_14  = [ 2.976196e+00 ; 4.983609e+00 ; 2.909160e+00 ];

%-- Image #15:
omc_15 = [ 2.102262e+00 ; 2.110838e+00 ; -2.395156e-01 ];
Tc_15  = [ -1.280229e+02 ; -1.145592e+02 ; 7.953767e+02 ];
omc_error_15 = [ 1.483208e-02 ; 1.642774e-02 ; 3.167945e-02 ];
Tc_error_15  = [ 6.652152e+00 ; 1.038372e+01 ; 5.929063e+00 ];

%-- Image #16:
omc_16 = [ 2.764909e+00 ; 6.433621e-04 ; 3.647849e-02 ];
Tc_16  = [ -1.122591e+02 ; 1.106422e+02 ; 4.864949e+02 ];
omc_error_16 = [ 1.212851e-02 ; 3.589135e-03 ; 1.742182e-02 ];
Tc_error_16  = [ 4.138407e+00 ; 6.876403e+00 ; 3.875283e+00 ];

%-- Image #17:
omc_17 = [ 1.668102e+00 ; 1.647715e+00 ; -8.169808e-01 ];
Tc_17  = [ -1.129804e+02 ; -6.040201e+01 ; 3.862644e+02 ];
omc_error_17 = [ 8.181424e-03 ; 9.403682e-03 ; 1.123340e-02 ];
Tc_error_17  = [ 3.213577e+00 ; 4.990539e+00 ; 2.813227e+00 ];

%-- Image #18:
omc_18 = [ 7.144678e-01 ; 2.176372e+00 ; -1.548199e+00 ];
Tc_18  = [ -2.299868e+01 ; -5.501387e+01 ; 5.488192e+02 ];
omc_error_18 = [ 7.309152e-03 ; 1.318904e-02 ; 1.389443e-02 ];
Tc_error_18  = [ 4.576812e+00 ; 7.243354e+00 ; 3.243756e+00 ];

%-- Image #19:
omc_19 = [ 4.010797e-02 ; -2.957237e+00 ; 8.675032e-01 ];
Tc_19  = [ 1.279422e+02 ; -1.520459e+02 ; 5.893858e+02 ];
omc_error_19 = [ 5.306683e-03 ; 1.050554e-02 ; 1.598491e-02 ];
Tc_error_19  = [ 4.973178e+00 ; 7.487237e+00 ; 4.525269e+00 ];

%-- Image #20:
omc_20 = [ 1.092142e+00 ; -1.798417e+00 ; 1.321616e+00 ];
Tc_20  = [ 1.666472e+02 ; 2.189857e+01 ; 4.029969e+02 ];
omc_error_20 = [ 8.085590e-03 ; 1.256906e-02 ; 1.200458e-02 ];
Tc_error_20  = [ 3.367057e+00 ; 5.540257e+00 ; 3.181298e+00 ];

%-- Image #21:
omc_21 = [ 1.307517e+00 ; 2.365966e+00 ; -7.337646e-01 ];
Tc_21  = [ -2.480070e+01 ; -3.630467e+01 ; 9.295035e+02 ];
omc_error_21 = [ 9.100305e-03 ; 1.404044e-02 ; 2.122829e-02 ];
Tc_error_21  = [ 7.750075e+00 ; 1.245729e+01 ; 7.276509e+00 ];

%-- Image #22:
omc_22 = [ 2.166024e+00 ; 2.161427e+00 ; -1.923897e-01 ];
Tc_22  = [ -1.293688e+02 ; -8.852543e+01 ; 3.988724e+02 ];
omc_error_22 = [ 7.314904e-03 ; 8.212380e-03 ; 1.575201e-02 ];
Tc_error_22  = [ 3.344433e+00 ; 5.221390e+00 ; 2.975702e+00 ];

%-- Image #23:
omc_23 = [ 2.013307e+00 ; 1.890509e-02 ; 1.497457e-02 ];
Tc_23  = [ -9.072468e+01 ; 4.476414e+01 ; 2.301306e+02 ];
omc_error_23 = [ 1.228317e-02 ; 5.424350e-03 ; 8.540150e-03 ];
Tc_error_23  = [ 1.957944e+00 ; 3.222943e+00 ; 1.849078e+00 ];

%-- Image #24:
omc_24 = [ 8.327110e-01 ; -2.042279e+00 ; 1.539531e+00 ];
Tc_24  = [ 1.647422e+02 ; 7.236329e+00 ; 4.177369e+02 ];
omc_error_24 = [ 7.214797e-03 ; 1.338870e-02 ; 1.320546e-02 ];
Tc_error_24  = [ 3.479768e+00 ; 5.691923e+00 ; 3.069884e+00 ];

%-- Image #25:
omc_25 = [ 2.383564e+00 ; -1.025542e+00 ; 4.186388e-01 ];
Tc_25  = [ 1.265413e+02 ; 1.559704e+02 ; 5.632092e+02 ];
omc_error_25 = [ 1.118166e-02 ; 8.432000e-03 ; 1.501865e-02 ];
Tc_error_25  = [ 4.796650e+00 ; 8.065015e+00 ; 4.620373e+00 ];

%-- Image #26:
omc_26 = [ 2.460042e+00 ; 9.671717e-01 ; -2.492216e-01 ];
Tc_26  = [ -1.577490e+02 ; 3.903558e+00 ; 8.472538e+02 ];
omc_error_26 = [ 1.397586e-02 ; 8.177699e-03 ; 1.946585e-02 ];
Tc_error_26  = [ 7.087841e+00 ; 1.138109e+01 ; 7.087662e+00 ];

%-- Image #27:
omc_27 = [ 2.000677e+00 ; -6.217752e-01 ; 3.992769e-01 ];
Tc_27  = [ -1.268149e+01 ; 6.086246e+01 ; 2.494317e+02 ];
omc_error_27 = [ 1.158087e-02 ; 6.860314e-03 ; 9.321880e-03 ];
Tc_error_27  = [ 2.115050e+00 ; 3.535833e+00 ; 1.959074e+00 ];

%-- Image #28:
omc_28 = [ 1.865950e+00 ; -1.795182e+00 ; 6.173960e-01 ];
Tc_28  = [ 1.328021e+02 ; 9.849965e+00 ; 6.060992e+02 ];
omc_error_28 = [ 8.060773e-03 ; 1.016683e-02 ; 1.431000e-02 ];
Tc_error_28  = [ 5.068906e+00 ; 8.239179e+00 ; 4.603373e+00 ];

%-- Image #29:
omc_29 = [ 2.447115e+00 ; 6.655385e-01 ; -1.317135e-01 ];
Tc_29  = [ -7.047164e+01 ; 1.083582e+01 ; 4.326477e+02 ];
omc_error_29 = [ 1.047017e-02 ; 4.706765e-03 ; 1.256082e-02 ];
Tc_error_29  = [ 3.625752e+00 ; 5.812678e+00 ; 3.276744e+00 ];

%-- Image #30:
omc_30 = [ 1.910357e+00 ; -1.785800e+00 ; 5.036252e-01 ];
Tc_30  = [ 1.644596e+02 ; 1.230238e+02 ; 1.186549e+03 ];
omc_error_30 = [ 1.441030e-02 ; 1.534836e-02 ; 2.549100e-02 ];
Tc_error_30  = [ 1.000798e+01 ; 1.635683e+01 ; 1.161932e+01 ];

%-- Image #31:
omc_31 = [ 1.824836e+00 ; 6.940609e-01 ; -4.431607e-01 ];
Tc_31  = [ -1.478533e+02 ; -2.669278e+01 ; 3.774357e+02 ];
omc_error_31 = [ 1.207379e-02 ; 7.475972e-03 ; 8.678940e-03 ];
Tc_error_31  = [ 3.146215e+00 ; 4.994900e+00 ; 2.926412e+00 ];

%-- Image #32:
omc_32 = [ 1.500170e+00 ; 1.411863e+00 ; -9.572675e-01 ];
Tc_32  = [ -1.307357e+02 ; -9.475876e+00 ; 4.194353e+02 ];
omc_error_32 = [ 9.582920e-03 ; 1.012343e-02 ; 1.038064e-02 ];
Tc_error_32  = [ 3.482808e+00 ; 5.610294e+00 ; 2.770164e+00 ];

%-- Image #33:
omc_33 = [ 1.713220e+00 ; -1.676471e+00 ; 7.997114e-01 ];
Tc_33  = [ 1.262574e+02 ; 7.294582e+00 ; 5.288213e+02 ];
omc_error_33 = [ 8.410514e-03 ; 1.046819e-02 ; 1.269905e-02 ];
Tc_error_33  = [ 4.417755e+00 ; 7.182610e+00 ; 3.960753e+00 ];

%-- Image #34:
omc_34 = [ 2.545046e+00 ; -8.178145e-01 ; 1.797357e-01 ];
Tc_34  = [ -3.753674e+00 ; 1.281633e+02 ; 7.792258e+02 ];
omc_error_34 = [ 1.495127e-02 ; 7.513549e-03 ; 2.101376e-02 ];
Tc_error_34  = [ 6.578243e+00 ; 1.087185e+01 ; 6.961739e+00 ];

%-- Image #35:
omc_35 = [ 2.066876e+00 ; -7.103926e-01 ; 3.710652e-01 ];
Tc_35  = [ -2.483645e+01 ; 5.533116e+01 ; 2.991729e+02 ];
omc_error_35 = [ 1.126683e-02 ; 6.768399e-03 ; 9.869598e-03 ];
Tc_error_35  = [ 2.517471e+00 ; 4.190297e+00 ; 2.251490e+00 ];

%-- Image #36:
omc_36 = [ 2.324734e-02 ; -3.006734e+00 ; 7.921388e-01 ];
Tc_36  = [ 1.254002e+02 ; -1.199357e+02 ; 9.561915e+02 ];
omc_error_36 = [ 6.559892e-03 ; 1.560573e-02 ; 2.300229e-02 ];
Tc_error_36  = [ 8.004721e+00 ; 1.253334e+01 ; 7.774912e+00 ];

