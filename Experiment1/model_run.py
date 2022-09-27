import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
import scipy.ndimage
from scipy.interpolate import interp1d
import car
import track
import matplotlib.cm as cm
from scipy import stats
import subprocess
import os
from numba import jit
import scipy.optimize


@jit(nopython=True)
def find(arr, pos):
    arr = arr - pos
    arr = (arr[:,0]**2 + arr[:,1]**2)**0.5
    return np.argmin(arr)


data = pd.read_feather("./non_model_data/data.feather")
data[data.ID == 203] = 503 #participant partiaööy labelled wrong, actually 503
"""
launch_ind = [False for i in range(len(data))]
launch = data['launch']
for i, row in data.iterrows():
    if row.launch != -100:
        launch_ind[row.launch] = True


data['launching'] = launch_ind

data.to_feather("data_fix_partial3.feather")
"""
#landing = data['landing']
#launch = data['launch']
#launching = data['launcing']

#locs = np.where(landing)[0]
#prev = 0
#for loc in locs:
#    launch[prev:loc+1] = launch[loc]    
#    prev = loc + 1

ordata = data
cond = 1
data = data[data.condition == cond]
data = data[data.drivingmode == 0]







middle = np.array(track.TrackMaker(10000, 25)[0])
inneredge = np.array(track.TrackMaker(10000, 25-1.5)[0])
outeredge = np.array(track.TrackMaker(10000, 25+1.5)[0])
narrow = track.getObstacles(0)
wide = track.getObstacles(1)

#fig0, axes = plt.subplots(1,3, sharex = True, sharey = True)
fig0, fig0b, fig0c = plt.figure(figsize=(4, 6)), plt.figure(figsize=(4, 6)), plt.figure(figsize=(4, 6)) #axes[0], axes[1], axes[2]
ax1, ax2, ax3 = fig0.gca(), fig0b.gca(), fig0c.gca()
fig0, axes = plt.subplots(2,2, sharex = True, sharey = True, figsize=(9, 12))
ax0, ax1, ax2, ax3 = axes[0,0], axes[0,1],  axes[1,0], axes[1,1]
#ax1.plot(middle[:,0], middle[:,1], '-k', linewidth = 0.2)
ax0.plot(inneredge[:,0], inneredge[:,1], '-k', linewidth = 0.2)
ax0.plot(outeredge[:,0], outeredge[:,1], '-k', linewidth = 0.2)

ax1.plot(inneredge[:,0], inneredge[:,1], '-k', linewidth = 0.2)
ax1.plot(outeredge[:,0], outeredge[:,1], '-k', linewidth = 0.2)

ax2.plot(inneredge[:,0], inneredge[:,1], '-k', linewidth = 0.2)
ax2.plot(outeredge[:,0], outeredge[:,1], '-k', linewidth = 0.2)

ax3.plot(inneredge[:,0], inneredge[:,1], '-k', linewidth = 0.2)
ax3.plot(outeredge[:,0], outeredge[:,1], '-k', linewidth = 0.2)


if cond % 2 == 0:
    obs = narrow
else:
    obs = wide

for i, t in enumerate(obs):
    t = list(t)
    circle = plt.Circle((t[0], t[2]),0.5, color = 'blue', alpha = 0.8)
    ax0.add_patch(circle)
    circle = plt.Circle((t[0], t[2]),0.5, color = 'blue', alpha = 0.8)
    ax1.add_patch(circle)
    circle = plt.Circle((t[0], t[2]),0.5, color = 'blue', alpha = 0.8)
    ax2.add_patch(circle)
    circle = plt.Circle((t[0], t[2]),0.5, color = 'blue', alpha = 0.8)
    ax3.add_patch(circle)

#ax2.set_aspect("equal")
#ax1.set_aspect("equal")
realcar_plot, = ax1.plot(np.nan, np.nan, 'og')
simcar_plot, = ax2.plot(np.nan,np.nan, 'or')

constcar_plot, = ax3.plot(np.nan,np.nan, 'oy')

gaze_plot, = ax2.plot(np.nan,np.nan, '.r')
const_plot, = ax3.plot(np.nan,np.nan, '.r')


mean_path = np.load("./non_model_data/mean_path.npy")
middle[:,0] = scipy.ndimage.gaussian_filter(middle[:,0], 50)
middle[:,1] = scipy.ndimage.gaussian_filter(middle[:,1], 50)

l0, l1, l2, l3 = 62000 - 2530, 65999 + 7030, 25999 + 7030, 22000 - 2530
data = data[((data.road_ind >= l0) & (data.road_ind <= l1))|((data.road_ind >= l3) & (data.road_ind <= l2))]
#data = data[data.roadsection == 2]
data = data.sort_values(by = ['ID', 'block', 'currtime'])
#minp = 18675
#maxp = 27999
#middle[minp:maxp,0] = mean_path[:,0]
#middle[minp:maxp,1] = mean_path[:,1]

middle = mean_path
#25999 22000 24000

#16500.0
#28499.0
#plt.plot(mean_path[25999 - 16500,0], mean_path[25999 - 16500,1], 'og')
#plt.plot(mean_path[22000 - 16500,0], mean_path[22000 - 16500,1], 'or')
#plt.plot(mean_path[24000 - 16500,0], mean_path[24000 - 16500,1], 'ob')
#plt.plot(mean_path[:,0], mean_path[:,1])
#plt.show()

middle[:,0] = scipy.ndimage.gaussian_filter(middle[:,0], 50)
middle[:,1] = scipy.ndimage.gaussian_filter(middle[:,1], 50)



d = 0
prev = middle[0]
dist = []
for m in middle:
    d += np.linalg.norm(m - prev)
    dist.append(d)
    prev = m

realcar_plot, = ax1.plot(np.nan, np.nan, 'og')
simcar_plot, = ax2.plot(np.nan,np.nan, 'or')
constcar_plot, = ax3.plot(np.nan,np.nan, 'oy')

#print(len(middle))
#f = scipy.interpolate.interp1d(dist, middle, axis = 0)
#middle = f(np.linspace(0,d,int(d*100)))
#print(len(middle))

#middle = np.array(track.TrackMaker(10000, 25)[0])[minp:maxp]

#ax3.plot(middle[0,0], middle[0,1], 'ok')
p_row = None

car_sim = car.Car(0,0)
car_real = car.Car(0,0)
steer_point = None
dt = 1/60

def get_target_interpolators():

    r = 0
    interpolators = []
    interpolators_gaze = []
    target = []
    steer_points = []
    p_row = None
    gaze = []
    participant = 0
    last = (data.index[-1])
    partn = 0

    for i, row in data.iterrows():
        #rint(row.confidence)
        if type(p_row) == type(None) or abs(row.currtime - p_row.currtime) > 1 or i == last:
            car_const = car.Car(row.posx, row.posz, -np.deg2rad(row.yaw))

            if(row.ID != participant and participant != 0):
                #print(participant, partn)
                partn = 0
            partn += 1

            if len(target) > 1:
                target = np.array(target)
                c = (np.abs(target[:,3]) < (89/2)) & (np.abs(target[:,4]) < (58/2))
                target = target[c]

                f = scipy.interpolate.interp1d(target[:,0][target[:,-1] > 0.8], target[:,1:3][target[:,-1] > 0.8], axis = 0, fill_value = 'extrapolate', bounds_error = False)
                interpolators.append([f, participant])

                f2 = scipy.interpolate.interp1d(target[:,0][target[:,-1] > 0.8], target[:,3][target[:,-1] > 0.8], fill_value = 'extrapolate', bounds_error = False)
                interpolators_gaze.append([f2, participant])
            else:
                interpolators.append(False)
                interpolators_gaze.append(False)
            target = []
            steer_points = []
            gaze = []
            r += 1
    
        pos_real = np.array([abs(row.posx), abs(row.posz)])
        cr = np.linalg.norm(middle - pos_real, axis = 1)
        closest = np.argmin(cr)

        p_row = row

        sign = np.sign(car_const.x)

        sign_gx = np.sign(row.hangle)

        #gaze.append([closest, row.hangle])


        x,y = row.xpog*sign, row.zpog*sign
        gx,gy = row.hangle, row.vangle

        target.append([closest, x, y, gx,gy, row.confidence])
        




    return interpolators, interpolators_gaze

r = 0
p_row = None

interpolators, interpolators_gaze = get_target_interpolators()


    

x = np.arange(len(mean_path)) 
mean_interp = []
mean_g_interp = []

for f,f2 in zip(interpolators, interpolators_gaze):
    if type(f) == bool:
        continue
    f, f2 = f[0], f2[0]
    d = f(x)    
    
    mean_interp.append(d.tolist())

    d = f2(x)    
    
    mean_g_interp.append(d.tolist())

mean_interp = np.array(mean_interp)
mean_g_interp = np.array(mean_g_interp)


mean_interp = np.nanmean(mean_interp, axis = 0)
mean_g_interp = np.nanmean(mean_g_interp, axis = 0)



def performance(mean_diff_sim, yaw_rates, participant, mean_diff_real):
    mean_diff_sim = np.array(mean_diff_sim)
    yaw_rates = np.array(yaw_rates)
    int_range = [22000 - 16500, 24000 - 16500,  25999 - 16500]
    #print(mean_diff_sim)

    #mean_diff_real = np.load("actual_traj.npy")

    if participant >= 0:
        #print(participant)
        c = (mean_diff_real[:,-1] == participant)
        mean_diff_real = mean_diff_real[c]
        mean_diff_sim  = mean_diff_sim[c]
        yaw_rates = yaw_rates[c]

    scatter = [[],[],[]]
    yaw1 = []
    yaw2 = []
    results = []
    for part in np.unique(mean_diff_real[:,-1]):
        c = (mean_diff_real[:,-1] == part)
        for r in np.unique(mean_diff_sim[c,0]):
            if r == -100:
                continue
            c = (mean_diff_real[:,0] == r) & (np.isfinite(mean_diff_real[:,0])) & (mean_diff_real[:,-1] == part)
            
            #min_i, max_i = np.min(mean_diff_real[c,1]), np.max(mean_diff_real[c,1])
            #print(max_i, min_i, np.sum(c))
            inds = np.arange(3000, 10500)

            if np.sum(c) == 0:
                continue
            #print(np.min(mean_diff_sim[c,1]))
            #print(np.max(mean_diff_sim[c,1]))
            fr = scipy.interpolate.interp1d(mean_diff_real[c,1], mean_diff_real[c,3:5], axis = 0, fill_value = np.nan, bounds_error = False)
            f = scipy.interpolate.interp1d(mean_diff_sim[c,1], mean_diff_sim[c,3:5], axis = 0, fill_value = np.nan, bounds_error = False)
            results += np.linalg.norm(fr(inds) - f(inds), axis = 1).tolist()

    r0 = np.nanmean(results)
    r1 = np.nanmedian(results)
    return r0



def search(params):
    if len(params) == 3:
        TH, smoothing, gain = params
    else:
        smoothing, gain = params
        TH = 0.5
    p_row = None
    path = ""

    #print(TH, smoothing, gain)
    car_sim = car.Car(0,0)
    car_real = car.Car(0,0)

    dt = 1/60

    include = []
    gaze_positions = []


    yaw_rates = []


    time = []
    run = []

    r = 0

    mean_diff_real = []
    mean_diff_sim = []
    mean_diff_const = []
    dists = []
    runs = []




    targets = np.array([obs[5], obs[4], obs[3], [25,0,25]])
    for i, row in data.iterrows():
        gazex, gazey = row.xpog, row.zpog
        sign = np.sign(row.posx)
        part = row.ID

        if type(p_row) == type(None) or abs(row.currtime - p_row.currtime) > 1:
            car_sim = car.Car(row.posx, row.posz, -np.deg2rad(row.yaw))
            car_sim.rotdt = np.deg2rad(row.yawrate)


            until, until2 = 0, 0
            last, last2 = len(middle), len(middle)


            r += 1
            interp = interpolators[r][0]
            interp_g = interpolators_gaze[r][0]

            time.append(row.currtime)
            yaw_rates.append([np.nan, np.nan])


            trial_t = row.currtime
            mean_diff_real.append([-100, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, part])   
            mean_diff_sim.append([-100, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, part])   
    
            obs1 = [obs[5], obs[4], obs[3], [25,0,25]]
            obstacle = list(obs1[0])
            obstacle = np.array([obstacle[0], obstacle[2]])

        if type(interp) == bool:
            p_row = row
            print("SKIPPING")
            continue


        rr = r

        car_real.x = row.posx
        car_real.y = row.posz

        sign = np.sign(car_real.x)

        if trial_t == row.currtime:
            pos_sim = np.array([abs(car_sim.x), abs(car_sim.y)])
            closest_sim = np.linalg.norm(middle - pos_sim, axis = 1)
            min_sim = np.argmin(closest_sim)

        pos_sim = np.array([abs(car_sim.x), abs(car_sim.y)])

        min_sim = find(middle[until:last], pos_sim) + until
        until = min_sim - 300 
        last = until + 600

        dist = np.linalg.norm(np.array([abs(car_sim.x), abs(car_sim.y)]) - obstacle)
        if dist < (TH*8) and len(obs1) > 1:
            obs1.pop(0)

        obstacle = list(obs1[0])
        obstacle = np.array([obstacle[0], obstacle[2]])

        if gaze_targ:
            target = interp(min_sim)
            target = [target[0]*sign, target[1]*sign]   
        else:
            target = obstacle
            target = np.array([target[0]*sign, target[1]*sign])




        pos_real = np.array([abs(row.posx), abs(row.posz)])
        closest_real = np.linalg.norm(middle[until2:last2] - pos_real, axis = 1)
        min_real =  np.argmin(closest_real) + until2
        until2 = min_real - 300 
        last2 = until2 + 600

        gaze = row.hangle, row.vangle
        targ_real = row.xpog, row.zpog
        #print(targ_real, target)
        mean_diff_real.append([rr, min_real, row.currtime - trial_t, abs(row.posx), abs(row.posz),  np.nan, targ_real[0]*sign, targ_real[1]*sign, np.nan, gaze[0], gaze[1], -np.deg2rad(row.yaw), part])    
        mean_diff_sim.append([rr, min_sim, row.currtime - trial_t, abs(car_sim.x), abs(car_sim.y),  np.nan, target[0], target[1], np.nan, gaze[0], gaze[1], car_sim.rot, part])    
        #mean_diff_sim.append([rr, min_sim, row.currtime - trial_t, abs(car_sim.x), abs(car_sim.y), np.nan, car_sim.rot, part])    
        #mean_diff_real.append([rr, min_real, row.currtime - trial_t, pos_real[0], pos_real[1], 0, -np.deg2rad(row.yaw), part])
        yaw_rates.append([row.yawrate, np.rad2deg(car_sim.rotdt)])

        if pure:
            car_sim.pursuitsteer(target, dt)
        else:
            egoTarg = car.toEgo(target, car_sim.pos(), -car_sim.rot)
            hrad = np.arctan(egoTarg[0]/egoTarg[1]) 
            car_sim.goal = -hrad

        car_sim.speed = 8
        car_sim.goal *= gain

        car_sim.move(dt, smoothing)


            
        p_row = row



    mean_diff_sim =  np.array(mean_diff_sim)
    mean_diff_real = np.array(mean_diff_real)
    #plt.figure()
    #plt.plot(mean_diff_sim[:,3], mean_diff_sim[:,4], '.')
    #plt.show()


    yaw_rates = np.array(yaw_rates)

    controller = "Prop"
    if pure: controller = "Pure"
    targ = "WP"
    if gaze_targ: targ = "GP"

    if SAVE:
        np.save(f"./model_data/actual_traj.npy", mean_diff_real)
        np.save(f"./model_data/{controller}_{targ}_{TH}_{smoothing}_{gain}.npy", mean_diff_sim)
        np.save(f"./model_data/{controller}_{targ}_{TH}_{smoothing}_{gain}_yaw.npy", yaw_rates)
    errors = []
    for part in np.unique(mean_diff_sim[:,-1]):
        perf = performance(mean_diff_sim, yaw_rates, part, mean_diff_real)
        errors.append(perf)
    print(np.mean(errors))
    print(params, np.mean(errors))
    #full_res.append([TH, smoothing, gain, np.mean(errors)])
    return np.mean(errors)

SAVE = True
gaze_targ, pure = True, True
search([1.2360950028268918, 0.056903076171875025][::-1])
gaze_targ, pure = True, False
search([2.228062133789064,0.057962112426757834][::-1])
gaze_targ, pure = False, False
search([2.8314911166725327,0.15700592873609207, 0.2468805875866864][::-1])
gaze_targ, pure = False, True
search([1.248163704362749,0.05539396657280975, 0.5859206201200794][::-1])



