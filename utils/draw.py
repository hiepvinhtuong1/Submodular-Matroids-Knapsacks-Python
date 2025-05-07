import matplotlib.pyplot as plt
import numpy as np

# Dữ liệu từ draw.matlab
t = np.array([0.64, 0.68, 0.72, 0.76, 0.80, 0.84, 0.88, 0.92, 0.96, 1.00])

densitysearchsgs_arr = np.array([40.56642498130963, 44.6015168647238, 47.60950105311776, 49.64964858286437,
                                 51.8106083088252, 55.0432028507029, 58.60111195707734, 59.08506687854077,
                                 62.179635444985436, 65.82955330321249])
repeated_arr = np.array([46.236684877402126, 49.79459398377656, 50.44772557566571, 50.4922603166337,
                         53.81594792012029, 56.852886200921766, 59.59722889942855, 60.22408307137327,
                         63.78317969771177, 66.24133022571817])
greedy_arr = np.array([46.236684877402126, 49.79459398377656, 50.44772557566571, 50.4922603166337,
                       53.81594792012029, 56.852886200921766, 59.59722889942855, 60.22408307137327,
                       62.31436382328442, 66.24133022571817])
fantom_arr = np.array([48.12887666211999, 49.79459398377656, 50.52411362190529, 54.08164106920229,
                       58.240619655153004, 61.383778610304816, 61.59352937040007, 63.87672943247916,
                       65.8976921036234, 68.64203480213021])
sproutpp_arr = np.array([49.1673734019251495, 53.2341247494827206, 56.4285643303768212, 58.191942014152405,
                         60.8824191659898984, 61.2215953615274858, 63.621603229910265, 69.573127540831164,
                         73.912846467880525, 76.088277088554166])

plt.figure()
plt.plot(t, densitysearchsgs_arr, linewidth=2, label='DSSGS')
plt.plot(t, repeated_arr, linewidth=2, label='RP_Greedy')
plt.plot(t, greedy_arr, linewidth=2, label='Greedy')
plt.plot(t, fantom_arr, linewidth=2, label='FANTOM')
plt.plot(t, sproutpp_arr, linewidth=2, label='SPROUT++')

plt.xlim(0.64, 1.0)
plt.ylim(40, 80)
plt.xticks(t)
plt.xlabel('Knapsack budget', fontname='Times New Roman', fontsize=20)
plt.ylabel('Objective value', fontname='Times New Roman', fontsize=26)
plt.legend(loc='southeast')
plt.grid(True)

# Cài đặt font Times New Roman cho toàn bộ biểu đồ
plt.rc('font', family='Times New Roman')
plt.rc('axes', titlesize=14)
plt.rc('axes', labelsize=14)
plt.rc('xtick', labelsize=14)
plt.rc('ytick', labelsize=14)

plt.show()