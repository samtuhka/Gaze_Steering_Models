import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import car
from numba import jit
import json
import scipy.interpolate
import scipy.optimize
import math

@jit(nopython=True)
def find(arr, pos):
    arr = arr - pos
    arr = (arr[:,0]**2 + arr[:,1]**2)**0.5
    return np.argmin(arr)


def get_road():
    x = open('./non_model_data/road120_x.json')
    y = open('./non_model_data/road120_y.json')

    x = json.load(x)
    y = json.load(y)
      
    # Iterating through the json
    # list

    road = []
    for xr, yr in zip(x,y):
        road.append([xr,yr])

    road = np.array(road)
    road *= 50
    road[:,0] -= 800

    road2 = np.array(road)
    road2[:,1] *= -1
    return road, road2

xr = []
yr = []

xr_i = []
yr_i = []

xr_o = []
yr_o = []

s = 0
last_x, last_y = 0,0

def rotate(px, py, angle, ox, oy):

    qx = ox + np.cos(angle) * (px - ox) - np.sin(angle) * (py - oy)
    qy = oy + np.sin(angle) * (px - ox) + np.cos(angle) * (py - oy)
    return qx, qy




last_x = -50
for i in range(20):
    s = np.deg2rad(-30)
    e =  np.deg2rad(120) 
    rang = np.linspace(s, s + e, 1000)
    x = np.sin(rang)*50
    y = np.cos(rang)*50

    xi = np.sin(rang)*48.25
    yi = np.cos(rang)*48.25
    
    xo = np.sin(rang)*51.75
    yo = np.cos(rang)*51.75


    if i % 2 == 0:
        xi, yi = rotate(xi, yi, np.pi, 0, 0)
        xi, yi  = xi[::-1], yi[::-1]
        xo, yo = rotate(xo, yo, np.pi, 0, 0)
        xo, yo  = xo[::-1], yo[::-1]
        x, y = rotate(x, y, np.pi, 0, 0)
        x, y = x[::-1], y[::-1]
    #if i % 2 == 0:
    #    rang = np.linspace(np.pi - s, np.pi - s + np.deg2rad(12), 100)
    #    x = np.cos(rang)
    #    y = np.sin(rang)

    xo += last_x - x[0]
    yo += last_y - y[0]

    xi += last_x - x[0]
    yi += last_y - y[0]

    x += last_x - x[0]
    y += last_y - y[0]



    #if i % 2 == 0:
    #    x, y, = rotate(np.pi, x, y)
    #    x = x[::-1]
    #    y = y[::-1]

    #plt.plot(x[-1], y[-1], 'og')
    #plt.plot(x[0], y[0], 'or')

    last_x = x[-1]
    last_y = y[-1]
    xr += x.tolist()
    yr += y.tolist()

    if i % 2 == 0:
        xr_i += xi.tolist()
        yr_i += yi.tolist()

        xr_o += xo.tolist()
        yr_o += yo.tolist()
    else:
        xr_i += xo.tolist()
        yr_i += yo.tolist()

        xr_o += xi.tolist()
        yr_o += yi.tolist()

xr = np.array(xr)#*50 - 800
yr = np.array(yr)#*50
xr_i = np.array(xr_i)#*50 - 800
yr_i = np.array(yr_i)#*50
xr_o = np.array(xr_o)#*50 - 800
yr_o = np.array(yr_o)#*50
print(yr_i.shape, xr.shape)
road, road2 = get_road()
#np.save("mean_path2.npy", road)

#plt.plot(road[:,0], road[:,1], '.k')

#plt.show()

def get_interpolators(data):
    car_pos = np.array(data['car_pos'])
    interpolators = []
    for scenario in np.unique(data['scn_id']):
        c = data['scn_id'] == scenario
        turn = np.sign(car_pos[c][0,1])

        r = road
        if turn > 0:
            r = road2
        #plt.plot(r[:,0],r[:,1])
        #plt.plot(r[0,0],r[0,1], 'go')
        gaze_scen = np.array(data['gaze_landing'])[c]
        gazeS_scen = np.array(data['gaze_screen'])[c]
        car_scen = car_pos[c]
        conf =  np.array(data['confidence'])[c]
        #plt.plot(conf, '.')
        #plt.show()
        #plt.plot(gaze[:,0], gaze[:,2], 'or')
        #plt.plot(car[:,0], car[:,1], 'ob')
        #plt.plot(car[0,0], car[0,1], 'bx', markersize = 30)
        #plt.show()
        until = 0 
        last = int(0.1*len(r))
        target = []
    


        for i, car in enumerate(car_scen):
                

            min_sim = find(r[until:last], car) + until
            until = max(min_sim - 300, 0) 
            last = until + 600
            target.append([min_sim, gaze_scen[i,0], gaze_scen[i,2], gazeS_scen[i,0], gazeS_scen[i,1], conf[i]])
            
        target = np.array(target)
        #filter gaze points out of the screen
        c = (np.abs(target[:,3]) < 35) & (np.abs(target[:,4]) < (35.0*(9/16.0)))
        target = target[c]
        try:
            f = scipy.interpolate.interp1d(target[:,0][target[:,-1] > 0.8], target[:,1:3][target[:,-1] > 0.8], axis = 0, fill_value = 'extrapolate', bounds_error = False)
            f2 = scipy.interpolate.interp1d(target[:,0][target[:,-1] > 0.8], target[:,3:5][target[:,-1] > 0.8], axis = 0, fill_value = 'extrapolate', bounds_error = False)
            interpolators.append([scenario, f, f2])
        except:
            interpolators.append([scenario, np.nan, np.nan])
    return interpolators       

#def main(params):
def main(gain, smooth, th, gaze_targ, pure):
    #gain, smooth = params
    #th = 0.5
    #gaze_targ, pure = True, True
    rootdir = "./birch20/"
    dt = 1/60

    #plt.plot(xr - 800, yr, '--k')
    #plt.plot(xr_i - 800, yr_i, 'b-')
    #plt.plot(xr_o - 800, yr_o, 'b-')

    #plt.plot(xr - 800, yr*-1, '--k')
    #plt.plot(xr_i - 800, yr_i*-1, 'b-')
    #plt.plot(xr_o - 800, yr_o*-1, 'b-')

    mean_diff_sim = []
    mean_diff_real = []
    yaw_rates = []
    sim_pos = []
    sign = 1
    rr = 0
    part = 0

    off = 0


    """
    for path in os.walk(rootdir):
            path = str(path[0]) + "/"
            kh = path.split("/")[-2]
            if os.path.exists(path + "world_timestamps.npy") and os.path.exists(path + "data_nonHide.pkl"):
                df = pd.read_pickle(path + "data_nonHide.pkl")
                part += 1
                print(path, part)
                gaze_scen = np.array(df['gaze_landing'])
                gazeS_scen = np.array(df['gaze_screen'])
                plt.plot(gaze_scen[:,0], gaze_scen[:,2], 'r.')
                #plt.plot(gazeS_scen[:,0], gazeS_scen[:,1], 'r.')
                plt.show()
    """

    #rootdir2 = "/media/samtuhka/Transcend/"   
    for part in range(1,17):

        df = pd.read_pickle(f"./non_model_data/data_part{part}.pkl")
        #gaze = np.load(path + 'screen_coords_bin.npy')


        yaw1 = np.array(df['yaw'])
        yaw2 = np.array(df['rotations'])

        #plt.plot(np.rad2deg(yaw1[:,3]), '-g')
        #plt.plot(df['timestamps'], np.rad2deg(-yaw1[:,1]), '-g')
        #plt.plot(df['timestamps'], np.rad2deg(yaw2[:,1]), '-k')
        #plt.plot(np.rad2deg(np.arctan(yaw1[:,0], yaw1[:,1])), '-b')
        #plt.show()
        #sfdsdfsdf

        #df = pd.read_pickle(path + "data_nonHideScreenClip.pkl")
        #df = pd.read_pickle(path + "data_nonHide_HorFix2Deg.pkl")

        #path2 = rootdir2 + path[2:]
        #marks1, marks2 = np.load(path2 + "calib_marks.npy", allow_pickle = True), np.load(path2 + "calib_marks_rect.npy", allow_pickle = True)
        #np.save(path + "calib_marks.npy", marks1)
        #np.save(path + "calib_marks_rect.npy", marks2)

        interpolators = get_interpolators(df)
        
        #wp2 = np.array(df['2nd_probes'])

        waypoints = np.array(df['1st_probes'])
        waypoints = waypoints[waypoints[:,2] < 0]
        _, ind = np.unique(waypoints[:,2], return_index = True)
        waypoints = waypoints[ind]
        waypoints = waypoints[np.argsort(waypoints[:,0])]
        waypoints[:,2] = np.abs(waypoints[:,2])


        car_pos = np.array(df['car_pos'])
        #car_pos[:,1] = np.abs(car_pos[:,1])
        sim_pos = []

        ts = df['timestamps']
        t = 0
        target = np.array([10000,10000])
        scn = 0
        scn_id = -1000
        for i in range(len(car_pos)):
            if df['scn_id'][i] != scn_id:
                trial_t = ts[i]

                #car_mark, = plt.plot(0, 0, 'ok')
                #gaze_mark, = plt.plot(0, 0, 'og')

                rr += 1

                if len(sim_pos) > 0:
                    
                    plt.plot(car_pos[c,0], car_pos[c,1], '-k', alpha = 0.1)

                    sim_pos = np.array(sim_pos)
                    c = np.isfinite(sim_pos[:,0])
                    sim_pos = sim_pos[c]
                    if len(sim_pos) == 0:
                        continue
                    #print(sign)                        

                    #plt.plot(sim_pos[0,0], sim_pos[0,1], 'or')
                    #plt.plot(sim_pos[:,0], sim_pos[:,1], '-g', alpha = 0.1)
                    #plt.xlim(min(sim_pos[:,0]), max(sim_pos[:,0]))
                    #plt.ylim(min(sim_pos[:,1]), max(sim_pos[:,1]))
                    #plt.show()

                sim_pos = []

                scenario = df['scn_id'][i]
                c = np.array(df['scn_id']) == scenario

                #wp1 = np.array(df['1st_probes'])[c]
                #_, ind = np.unique(wp1[:,0], return_index = True)
                #wp1 = wp1[ind].tolist()

                sign = np.sign(car_pos[c,1][0])
                        
                wp1 = np.array(waypoints)
                #plt.plot(wp1[:,0], wp1[:,2], 'og')
                wp1[:,2] *= sign
                #plt.plot(wp1[:,0], wp1[:,2], 'og')
                wp1 = wp1.tolist()


                #start rotation is weird? need to check webtrajsim code
                #print(sign, np.rad2deg(yaw2[i,1]))
                #print(sign, np.rad2deg(yaw1[i,1]))
                car_sim = car.Car(car_pos[i, 0], car_pos[i, 1], np.pi - yaw2[i,1])
                if sign > 0:
                    car_sim = car.Car(car_pos[i, 0], car_pos[i, 1], yaw2[i,1])

                #car_sim = car.Car(car_pos[i, 0], car_pos[i, 1], np.pi + np.deg2rad(6))
                #if sign > 0:
                #    car_sim = car.Car(car_pos[i, 0], car_pos[i, 1], np.pi*2 - np.deg2rad(12))

                #turn = np.sign(car_pos[i][1])
                f = interpolators[scn][1]
                r = road
                if sign > 0:
                    r = road2
                until = 0 
                last = int(0.1*len(r))

                until_g = 0 
                last_g = int(0.1*len(r))

                scn += 1
                scn_id = df['scn_id'][i]

                yaw_rates.append([np.nan, np.nan])
                mean_diff_sim.append([-100, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, part])   
                mean_diff_real.append([-100, np.nan, np.nan, np.nan, np.nan, np.nan,np.nan, np.nan, np.nan, part])     
            if type(f) == type(np.nan):
                yaw_rates.append([np.nan, np.nan])
                mean_diff_sim.append([-100, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,  np.nan, np.nan, np.nan, np.nan, np.nan, part])
                t = ts[i] 
                continue  
            dist = np.linalg.norm(np.array(car_sim.pos()) - target) 
            if (dist  < 10.5*th or dist > 100) and len(wp1) > 0:
                #plt.plot(target[0], target[1], 'og')
                target = wp1.pop(0)
                target = np.array([target[0], target[2]])

            #car_sim.x = car_pos[i,0]
            #car_sim.y = car_pos[i,1]

            min_sim = find(r[until:last], np.array([car_sim.x, car_sim.y])) + until
            until = max(min_sim - 100, 0) 
            last = until + 200
            
            if gaze_targ: target = f(min_sim)
            #print(target[1], car_sim.y)

            p2 =  np.array([target[0], (target[1])])
            pos = np.array([car_sim.x, (car_sim.y)])
            p2 = car.toEgo(p2, pos, -car_sim.rot)
            hrad = np.arctan(p2[0]/p2[1]) 
    
            #print(target)



            min_sim_g = find(r[until_g:last_g], np.array([target[0], target[1]])) + until_g
            until_g = max(min_sim_g - 100, 0) 
            last_g = until_g + 200

            #if d > 50:  
            #    plt.plot(target[0], target[1], 'or')
            #car_sim.goal = -hrad


            t = ts[i]


    
            pos_sim = np.array([car_sim.x, car_sim.y])
            pos_real = np.array([car_pos[i, 0], car_pos[i, 1]])

            d1 = np.linalg.norm(r[min_sim] - pos_sim)
            if d1 > 10:
                off += 1
                car_sim.x = np.nan
                car_sim.y = np.nan
            gaze = df['gaze_screen'][i]
            #print(gaze)
            mean_diff_real.append([rr, min_sim, ts[i] - trial_t, pos_real[0], pos_real[1], d1, target[0], target[1], car_sim.rot, part])    
            #mean_diff_sim.append([rr, min_sim, ts[i] - trial_t, pos_sim[0], pos_sim[1],  d1, target[0], target[1], gaze[0], gaze[1], car_sim.rot, part])
            mean_diff_sim.append([rr, min_sim, ts[i] - trial_t, pos_sim[0], pos_sim[1],  d1, target[0], target[1], min_sim_g, gaze[0], gaze[1], car_sim.rot, part])    
            yaw_rates.append([df['yawrate'][i], car_sim.rotdt])


            if pure: car_sim.pursuitsteer(target, dt)  
            else: car_sim.goal = -hrad
            
            car_sim.goal *= gain
            car_sim.speed = 10.5

            car_sim.move(dt,smooth)






    #np.save(f"actual_new.npy", mean_diff_real)
    #ssdfsdf
    #print(off/rr)
    mean_diff_real = np.load("./non_model_data/actual_traj.npy")
    mean_diff_sim = np.array(mean_diff_sim)
    s = mean_diff_sim[:,5]
    s = s[np.isfinite(s)]
    off = np.sum(s > 1.75)/len(s)
    controller = "Prop"
    if pure: controller = "Pure"

    targ = "WP"
    if gaze_targ: targ = "GP"
    
    np.save(f"./model_data/{controller}_{targ}_{th}_{smooth}_{gain}.npy", mean_diff_sim)
    np.save(f"./model_data/{controller}_{targ}_{th}_{smooth}_{gain}_yaw.npy", yaw_rates)

    error = []
    for r in np.unique(mean_diff_sim[:,0]):
        if r == -100:
            continue

        c = (mean_diff_real[:,0] == r) & (np.isfinite(mean_diff_real[:,0])) & (mean_diff_real[:,1] < 20000) #& (mean_diff_sim[:,1] < 20800)  & (mean_diff_real[:,1] < 20800)
        f = scipy.interpolate.interp1d(mean_diff_sim[c,1], mean_diff_sim[c,3:5], axis = 0,  fill_value = np.nan,  bounds_error = False)
        fr = scipy.interpolate.interp1d(mean_diff_real[c,1], mean_diff_real[c,3:5], axis = 0, fill_value = np.nan,  bounds_error = False)
        ind = mean_diff_real[c,1]
        err = np.linalg.norm(fr(ind) - f(ind), axis = 1)
        error.append(np.nanmean(err))
    error = np.mean(error)
    print(error, off, gain, smooth)

    #print("AAAA")
    plt.close('all')
    return error

#new opti
main(1.236095, 0.05690308, 0.52, True, True)
main(2.22806213,0.05796211, 0.52, True, False)
main(2.83149112,0.15700593, 0.24688059, False, False)
main(1.2481637,0.0553939, 0.58592062, False, True)
