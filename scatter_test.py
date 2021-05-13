import numpy as np
from io import StringIO 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d

# Zadanie 2

loaded_data = []

with open("GEFCOM.txt", "r") as file:
    for line in file:
        strintIO_line = StringIO(line)
        npLine = np.loadtxt(strintIO_line)
        loaded_data.append(npLine)

loaded_data = np.array(loaded_data)


poniedzialek_data = []
wtorek_data = []
sroda_data = []
czwarte_data = []
piatek_data = []
sobota_data = []
niedziela_data = []

for row in loaded_data:
    if(row[5]==1):
        poniedzialek_data.append(row)
    if(row[5]==2):
        wtorek_data.append(row)
    if(row[5]==3):
        sroda_data.append(row)
    if(row[5]==4):
        czwarte_data.append(row)
    if(row[5]==5):
        piatek_data.append(row)
    if(row[5]==6):
        sobota_data.append(row)
    if(row[5]==7):
        niedziela_data.append(row)

pon_data_len = len(poniedzialek_data)
wt_data_len = len(wtorek_data)
sr_data_len = len(sroda_data)
czw_data_len = len(czwarte_data)
pt_data_len = len(piatek_data)
sob_data_len = len(sobota_data)
ni_data_len = len(niedziela_data)

min_len = min([pon_data_len,wt_data_len,sr_data_len,czw_data_len,pt_data_len,sob_data_len,ni_data_len])

poniedzialek_data = np.array(poniedzialek_data)
wtorek_data = np.array(wtorek_data)
sroda_data = np.array(sroda_data)
czwarte_data = np.array(czwarte_data)
piatek_data = np.array(piatek_data)
sobota_data = np.array(sobota_data)
niedziela_data = np.array(niedziela_data)

# print(np.shape(niedziela_data))

# Cały tydzien

cala_data = np.zeros((7,min_len,6))

cala_data[0,:,:] = poniedzialek_data[:min_len,:]
cala_data[1,:,:] = wtorek_data[:min_len,:]
cala_data[2,:,:] = sroda_data[:min_len,:]
cala_data[3,:,:] = czwarte_data[:min_len,:]
cala_data[4,:,:] = piatek_data[:min_len,:]
cala_data[5,:,:] = sobota_data[:min_len,:]
cala_data[6,:,:] = niedziela_data[:min_len,:]

# Plotting

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

x_1 = []
for i in range(1,8):
    x_1.append(np.ones(24)*i)
x_1 = np.array(x_1)

y_1 = []
for i in range(1,25):
    y_1.append(np.ones(min_len)*i)

x_1 = np.linspace(1,7,24)
y_1 = np.linspace(1,24,24)

[m,n,s] = np.shape(cala_data)


for d in range(7):
    for h in range(24):
        ax.scatter(d,h,cala_data[d,h,2],c='orange')

# for i in range(int(n/24)):
#     plt.scatter(x_1,y_1,cala_data[:,:,2],c='orange')

# plt.scatter(x_1,y_1,cala_data[:,:,2])
# plt.scatter(x_1,y_1,cala_data[:,:,4])

# for i in range(7):
#     x_tyg = np.linspace(i*int(min_len/24),(i+1)*int(min_len/24)+1,int(min_len/24))
#     plt.plot(x_tyg,tydzien_c[i,:],label=lab_arr[i])
# plt.xlabel('Dzień tygodnia')
# plt.ylabel('Prognoza zapotrzebowania w całym systemie elektroenergetycznym')
# # plt.xticks(ticks=np.linspace(50,7*int(min_len/24),7)-50,labels=lab_arr)
# plt.title('Wykres sezonowy tygodniowy')
# plt.legend()
# # plt.savefig('List_3_zad_2_a.png', dpi=300)
plt.show()
