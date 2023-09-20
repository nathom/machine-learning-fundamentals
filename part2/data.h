#ifndef DATA_H
#define DATA_H

#include "matrix.h"

#define NUM_FEATURES 4
#define NUM_SAMPLES 100

static mfloat X_data[NUM_SAMPLES][NUM_FEATURES] = {
    {1.00468339e+03, 5.81843207e+01, 5.57484991e+01,
     9.16101798e+01},
    {1.12671840e+03, 1.83979969e+02, 2.02158606e+01,
     9.59115383e+01},
    {1.14072766e+03, 6.86204293e+01, 5.54728647e+01,
     6.72852588e+00},
    {1.28181963e+03, 8.05957789e+01, 5.33277094e+01,
     4.00576581e+01},
    {1.36390231e+03, 6.52927450e+01, 7.39015111e+01,
     1.15688505e+01},
    {1.58403022e+03, 7.92732631e+01, 6.21921326e+01,
     3.42303464e+01},
    {1.85869105e+03, 1.79214333e+02, 2.92535643e+01,
     1.77983105e+01},
    {1.94944834e+03, 1.67476268e+02, 3.14000873e+01,
     6.50379878e+01},
    {1.99801739e+03, 1.84808017e+02, 2.91444119e+01,
     3.94385700e+01},
    {2.09879159e+03, 2.10770932e+02, 2.72101041e+01,
     2.37100798e+01},
    {2.25398309e+03, 1.98184853e+02, 3.19611867e+01,
     2.73278546e+01},
    {2.33278237e+03, 7.49805933e+01, 9.07038626e+01,
     3.66741644e+01},
    {2.48190268e+03, 9.98106761e+01, 6.84789117e+01,
     2.67350364e+01},
    {2.48740245e+03, 1.64629445e+02, 4.44304243e+01,
     4.58708438e+01},
    {2.51461938e+03, 7.55754816e+01, 9.80747258e+01,
     4.19927916e+00},
    {2.64012479e+03, 1.34226611e+02, 5.53197194e+01,
     2.86965872e+01},
    {2.78958164e+03, 1.36912390e+02, 6.31625701e+01,
     4.17094654e+01},
    {2.81547282e+03, 1.37913729e+02, 6.55447191e+01,
     6.06754063e+01},
    {2.82755102e+03, 1.88138684e+02, 4.91075765e+01,
     2.78248066e+01},
    {2.87147497e+03, 2.35459576e+02, 4.00119746e+01,
     1.41853274e+01},
    {2.96596393e+03, 1.71413812e+02, 5.54365791e+01,
     8.54383758e+00},
    {2.96887798e+03, 3.53253033e+02, 2.91206445e+01,
     9.41126509e+01},
    {3.12386428e+03, 6.31002210e+01, 1.65710720e+02,
     4.22472667e+01},
    {3.19712970e+03, 2.34730253e+02, 4.78869789e+01,
     5.85350995e+01},
    {3.22188457e+03, 1.52177687e+02, 7.61918656e+01,
     9.19984786e+01},
    {3.28523873e+03, 1.82445468e+02, 6.39296926e+01,
     9.19209358e+00},
    {3.30461490e+03, 1.87425384e+02, 6.47237613e+01,
     8.77894871e+01},
    {3.42471100e+03, 1.58256708e+02, 7.66846583e+01,
     5.56071994e+01},
    {3.44919024e+03, 8.84097929e+01, 1.61407525e+02,
     1.73185919e+01},
    {3.47249614e+03, 1.10730855e+02, 1.33312561e+02,
     4.17142565e+01},
    {3.48399264e+03, 1.05747609e+02, 1.39865524e+02,
     7.79826261e+01},
    {3.64139357e+03, 1.58162828e+02, 9.51027906e+01,
     4.85566381e+01},
    {3.66070317e+03, 9.26205715e+01, 1.64779811e+02,
     9.85433190e+01},
    {3.66646155e+03, 1.65275704e+02, 9.53710475e+01,
     3.82971580e+01},
    {3.74303132e+03, 2.56103677e+02, 6.61293776e+01,
     7.52082517e+01},
    {3.91910519e+03, 1.74205669e+02, 9.90408595e+01,
     3.99059554e+01},
    {3.93886692e+03, 2.78670369e+02, 6.32877676e+01,
     8.30872578e+01},
    {4.17311971e+03, 1.12025396e+02, 1.62105081e+02,
     5.73390654e+01},
    {4.20668054e+03, 1.69706540e+02, 1.08143200e+02,
     7.28767113e+00},
    {4.33736427e+03, 2.49866502e+02, 7.37320737e+01,
     4.64536490e+00},
    {4.44634187e+03, 3.68002120e+02, 5.18158545e+01,
     1.42513598e+01},
    {4.55393366e+03, 2.05956630e+02, 9.35939917e+01,
     2.35352452e+00},
    {4.55635212e+03, 1.75761721e+02, 1.23171529e+02,
     8.46054697e+00},
    {4.63452554e+03, 9.46172073e+01, 2.29816362e+02,
     6.94797253e+01},
    {4.74858953e+03, 2.00679687e+02, 1.11289956e+02,
     5.39002812e+01},
    {4.81061333e+03, 2.16720686e+02, 1.04880160e+02,
     7.52411642e+01},
    {4.82640287e+03, 1.88007812e+02, 1.23030953e+02,
     9.14034095e+01},
    {4.95402852e+03, 1.12378225e+02, 2.08007153e+02,
     5.89298037e+01},
    {5.04505607e+03, 2.80677850e+02, 8.36426038e+01,
     7.28859636e+01},
    {5.04778720e+03, 8.12768958e+01, 2.90467341e+02,
     7.59510390e+01},
    {5.10881113e+03, 1.82688789e+02, 1.30224492e+02,
     3.84072044e+01},
    {5.27833201e+03, 1.23123647e+02, 1.98917617e+02,
     2.48673984e+01},
    {5.36346974e+03, 9.42353109e+01, 2.63591948e+02,
     2.12994875e+01},
    {5.68750834e+03, 1.42728843e+02, 1.86009176e+02,
     2.58928015e+01},
    {5.80680477e+03, 1.43633921e+02, 1.89960170e+02,
     2.81984479e+01},
    {5.85722919e+03, 1.99155636e+02, 1.41813080e+02,
     2.15155373e+01},
    {5.85856982e+03, 1.74039065e+02, 1.62898715e+02,
     8.79438461e+01},
    {6.02291802e+03, 2.67083793e+02, 1.08514780e+02,
     7.59429141e+01},
    {6.05119083e+03, 2.13746472e+02, 1.38499357e+02,
     5.64275031e+00},
    {6.06948015e+03, 2.42293194e+02, 1.23603352e+02,
     2.75985757e+01},
    {6.10930295e+03, 8.46923291e+01, 3.56663353e+02,
     3.19628946e+00},
    {6.13399577e+03, 2.48440732e+02, 1.27761435e+02,
     5.03183535e+01},
    {6.28076049e+03, 1.67857915e+02, 1.90309791e+02,
     4.81448590e+01},
    {6.32168009e+03, 2.00960093e+02, 1.61778575e+02,
     8.33057775e+01},
    {6.38978920e+03, 1.25156456e+02, 2.62958620e+02,
     3.14699467e+01},
    {6.43975641e+03, 1.18130466e+02, 2.79341301e+02,
     8.18221878e+01},
    {6.45363977e+03, 2.59369805e+02, 1.29310978e+02,
     9.68294028e+01},
    {6.46330823e+03, 1.06466174e+02, 3.18262892e+02,
     9.75241411e+00},
    {6.61918643e+03, 2.53306790e+02, 1.36534944e+02,
     7.93899661e+01},
    {6.82921108e+03, 2.69811665e+02, 1.29078601e+02,
     5.94056338e+01},
    {6.85969323e+03, 1.94430471e+02, 1.80765376e+02,
     4.85245507e+01},
    {6.99330121e+03, 2.03654232e+02, 1.73625144e+02,
     4.26330421e+01},
    {7.01957127e+03, 2.92261511e+02, 1.22756752e+02,
     7.86821581e+01},
    {7.23192430e+03, 1.34317990e+02, 2.70394284e+02,
     6.42967743e+01},
    {7.25306440e+03, 3.31123060e+02, 1.10490553e+02,
     8.06994221e+01},
    {7.25964478e+03, 1.01763362e+02, 3.66653058e+02,
     9.04119548e+01},
    {7.26673846e+03, 2.66703221e+02, 1.42368000e+02,
     6.21091075e+01},
    {7.39596601e+03, 2.76659949e+02, 1.40526766e+02,
     9.80658098e+01},
    {7.40207757e+03, 1.42620983e+02, 2.84983429e+02,
     6.12006971e+01},
    {7.41043531e+03, 2.04593319e+02, 2.00030529e+02,
     6.40277878e+01},
    {7.49556569e+03, 2.81684951e+02, 1.46002107e+02,
     5.59267452e+01},
    {7.54544796e+03, 1.95368852e+02, 2.12777293e+02,
     1.00092070e+01},
    {7.72588099e+03, 1.03203443e+02, 4.06089582e+02,
     7.29133062e+01},
    {7.79825269e+03, 9.83597321e+01, 4.31374442e+02,
     5.51971844e+01},
    {8.11157986e+03, 1.31871579e+02, 3.22852618e+02,
     4.56401343e+01},
    {8.58056297e+03, 2.16997336e+02, 1.97640585e+02,
     9.11366566e+01},
    {8.60080464e+03, 1.65931431e+02, 2.61788962e+02,
     3.04979857e+01},
    {8.64022442e+03, 1.77975229e+02, 2.45867642e+02,
     5.28366253e+01},
    {8.65034720e+03, 2.27407116e+02, 1.96436272e+02,
     7.00665453e+01},
    {8.92421055e+03, 2.88330702e+02, 1.54940067e+02,
     7.98507058e+01},
    {8.94952233e+03, 2.37841010e+02, 1.93469769e+02,
     4.64753340e+01},
    {9.06187236e+03, 9.94774091e+01, 4.71177395e+02,
     8.43670501e+01},
    {9.16145597e+03, 3.22020846e+02, 1.45726149e+02,
     7.71228564e+01},
    {9.23463708e+03, 2.37611395e+02, 1.99329268e+02,
     7.55736181e+00},
    {9.33670791e+03, 1.77894951e+02, 2.66335973e+02,
     5.54026537e+00},
    {9.33992979e+03, 1.42226771e+02, 3.34746047e+02,
     6.24597628e+01},
    {9.48568214e+03, 1.08341211e+02, 4.46430637e+02,
     3.53939274e+01},
    {9.65054822e+03, 2.59109612e+02, 1.87442806e+02,
     2.17039483e+01},
    {9.68729777e+03, 3.90678148e+02, 1.25517629e+02,
     5.83853435e+01},
    {9.74540886e+03, 1.23235072e+02, 3.98803045e+02,
     3.48147578e+01}};

static mfloat Y_data[NUM_SAMPLES] = {
    1322.24810598, 1262.2564561,  2117.00514996, 1064.61179894,
    1561.1033238,  828.29490924,  623.99861431,  1393.39588397,
    746.80198847,  1506.20555099, 551.90974926,  999.4812617,
    946.14713433,  1048.22223963, 3074.97130598, 976.61621927,
    903.9276467,   1189.68041717, 1659.43610427, 1553.69735781,
    2249.28837603, 1575.48177269, 1231.19664207, 1736.00776209,
    1087.93008652, 2499.10326746, 1182.26365181, 1358.03314576,
    1620.2394242,  1236.70873763, 1420.99512457, 1310.41902511,
    973.33245648,  1546.40303639, 1812.94601238, 1597.62339866,
    954.21474487,  1375.00937517, 2813.5281454,  3347.01456132,
    1997.68835237, 4451.40359539, 2911.10130456, 1795.32238101,
    1879.38302095, 1162.42267269, 2330.65933856, 2202.93431468,
    1888.28332452, 2068.66487244, 2198.6895092,  2793.69605848,
    2586.62857862, 2464.65543967, 2024.38004226, 2776.5602624,
    1876.68795169, 2108.97758005, 4483.65458005, 2596.65205709,
    4898.15248099, 2238.21871752, 2744.88499253, 2328.60788747,
    2968.4145176,  2203.45197715, 3158.59009031, 3431.75888127,
    2710.92299539, 2955.55885567, 3136.78487105, 2459.73850345,
    2445.48337713, 2837.73383413, 2554.5949605,  2490.04581654,
    3320.0030796,  2656.06938695, 3040.68994919, 3278.52004215,
    3589.67482982, 3875.84648908, 2717.13276853, 3227.08762788,
    3004.52181731, 3270.37498043, 3414.0957463,  3245.48446539,
    3700.43255489, 3645.8417073,  3128.68312686, 3384.53337849,
    3786.2129267,  4537.89731565, 5428.03814118, 3523.2074019,
    3552.06763795, 3695.28780002, 3715.69750062, 3256.61587167};

static matrix X = {.buf = (mfloat *)X_data,
                   .rows = NUM_SAMPLES,
                   .cols = NUM_FEATURES};
static matrix Y = {
    .buf = (mfloat *)Y_data, .rows = 1, .cols = NUM_SAMPLES};

#endif
