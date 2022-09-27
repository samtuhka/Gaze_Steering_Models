import numpy as np
import scipy.interpolate
import scipy.stats
import os
import matplotlib.pyplot as plt
import scipy.ndimage
#import track
import matplotlib as mpl
from scipy.stats.kde import gaussian_kde
import warnings
warnings.filterwarnings("ignore")
import track


folder = "./model_data/"
true_traj = np.load(f"{folder}/actual_traj.npy")

mean_path = np.load("./non_model_data/mean_path.npy")
mean_path[:,0] = scipy.ndimage.gaussian_filter(mean_path[:,0], 50)
mean_path[:,1] = scipy.ndimage.gaussian_filter(mean_path[:,1], 50)
inneredge = np.array(track.TrackMaker(10000, 25-1.5)[0])
outeredge = np.array(track.TrackMaker(10000, 25+1.5)[0])
indices = np.arange(len(mean_path))

comp_fig, comp_axes, wp_grid = None, None, None

def plot(title = ""):
    string = title
    title = ""
    if 'Prop' in string:
        title += 'Proportional controller with '
    else:
        title += 'Pure Pursuit controller with '
    if 'G' in string:
        title += 'gaze input'
    else:
        title += "waypoint input"

    #fig0, axes = plt.subplots(2,1, sharex = True, sharey = True, figsize=(9, 12))
    #fig0.suptitle(title)
    #ax1, ax3 = axes[0], axes[1]
    #ax1.arrow(22,55,0,-10, head_width=1, head_length=1, fc='k', ec='k')
    #plt.show()


    #fig1, axes = plt.subplots(1,4, sharex = True, sharey = True, figsize=(10, 10))
    fig0 = plt.figure(figsize=(10, 5))
    grid = fig0.add_gridspec(1, 4)
    axes = [fig0.add_subplot(grid[i]) for i in range(3)]
    
    hist_grid = grid[3].subgridspec(3, 1, hspace = 0.35)
    wp_grid_temp = hist_grid.subplots()

    ax0, ax2 = axes[0], axes[1]

    axes[0].arrow(22, 60, 0, -5, head_width=0.4, linewidth = 2, head_length=0.8, fc='k', ec='k')
    axes[1].arrow(22, 60, 0, -5, head_width=0.4, linewidth = 2, head_length=0.8, fc='k', ec='k')
    axes[2].arrow(22, 60, 0, -5, head_width=0.4, linewidth = 2, head_length=0.8, fc='k', ec='k')

    #Eaxes[1,1].axis('off')

    if "Pure" in title:
        global comp_fig, comp_axes, wp_grid
        comp_fig = fig0
        comp_axes = axes
        wp_grid = wp_grid_temp
    else:
        fig0 = comp_fig
        axes = comp_axes
        ax2 = axes[2]


    ax0.set_xlabel('x-coord (m)')
    ax2.set_xlabel('x-coord (m)')
    ax0.set_ylabel('y-coord (m)')
    ax2.set_ylabel('y-coord (m)')

    fig1, axes = plt.subplots(1,2, sharex = True, sharey = True, figsize=(4, 5))
    fig1.suptitle(title)
    ax1, ax3 = axes
    #ax0, ax1, ax2, ax3 = axes[0,0], axes[0,1],  axes[1,0], axes[1,1]
    #ax1.plot(middle[:,0], middle[:,1], '-k', linewidth = 0.2)
    ax0.plot(inneredge[:,0], inneredge[:,1], '-k', linewidth = 0.2)
    ax0.plot(outeredge[:,0], outeredge[:,1], '-k', linewidth = 0.2)

    ax1.plot(inneredge[:,0], inneredge[:,1], '-k', linewidth = 0.2)
    ax1.plot(outeredge[:,0], outeredge[:,1], '-k', linewidth = 0.2)

    ax2.plot(inneredge[:,0], inneredge[:,1], '-k', linewidth = 0.2)
    ax2.plot(outeredge[:,0], outeredge[:,1], '-k', linewidth = 0.2)

    ax3.plot(inneredge[:,0], inneredge[:,1], '-k', linewidth = 0.2)
    ax3.plot(outeredge[:,0], outeredge[:,1], '-k', linewidth = 0.2)

    fig2, ax_yaw = plt.subplots(1,1, sharex = True, sharey = True, figsize = (8,4))
    fig2.suptitle(title)

    fig3, gaze_ax = plt.subplots(1,1, sharex = True, sharey = True, figsize = (4,6))
    #fig3.suptitle("Gaze distribution")

    gaze_ax.plot(inneredge[:,0], inneredge[:,1], '-k', linewidth = 0.2)
    gaze_ax.plot(outeredge[:,0], outeredge[:,1], '-k', linewidth = 0.2)



    return fig0, fig2, fig3, (ax0, ax1, ax2, ax3, ax_yaw, gaze_ax)


def fin_plots(fig0, fig1, fig2, axes, title = ""):
    string = title
    title = ""
    if 'Prop' in string:
        title += 'Proportional controller with '
    else:
        title += 'Pure pursuit controller with '
    if 'G' in string:
        title += 'gaze input'
    else:
        title += "waypoint input"

    path = "./figures/Trout_" + string + "_"


    ax0, ax1, ax2, ax3, ax_yaw, ax_gaze = axes
    ax0.set_title('Human trajectories', fontsize = 8)
    ax0.set_xlim(20, 30)
    ax0.set_ylim(30, 65) 

    ax1.set_title('Human trajectories (part-means)')
    ax1.set_xlim(20, 30)
    ax1.set_ylim(30, 65)

    ax2.set_title('Model trajectories\n'+title, fontsize = 8)
    ax2.set_xlim(20, 30)
    ax2.set_ylim(30, 65)

    ax3.set_title('Model trajectories (part-means)')
    ax3.set_xlim(20, 30)
    ax3.set_ylim(30, 65)

    ax_gaze.set_title('Gaze distribution in experiment 1')
    ax_gaze.set_xlim(20, 30)
    ax_gaze.set_ylim(30, 65)
    ax_gaze.arrow(22, 60, 0, -5, head_width=0.4, linewidth = 2, head_length=0.8, fc='k', ec='k')


    aspect = 0.7
    ax0.set_aspect(aspect)
    ax1.set_aspect(aspect)
    ax2.set_aspect(aspect)
    ax3.set_aspect(aspect)
    ax_gaze.set_aspect(aspect)

    #ax0.plot(mean_path[:,0], mean_path[:,1], '-k', linewidth = 1)
    #ax1.plot(mean_path[:,0], mean_path[:,1], '-k', linewidth = 1)
    #ax2.plot(mean_path[:,0], mean_path[:,1], '-k', linewidth = 1)
    #ax3.plot(mean_path[:,0], mean_path[:,1], '-k', linewidth = 1)

    obs = track.getObstacles(1)
    for i, t in enumerate(obs):
        t = list(t)

        circle = plt.Circle((t[0], t[2]),0.5, color = 'blue', alpha = 0.8, fill = False, zorder = 10)
        ax0.add_patch(circle)
        circle = plt.Circle((t[0], t[2]),0.5, color = 'blue', alpha = 0.8, fill = False, zorder = 10)
        ax1.add_patch(circle)
        circle = plt.Circle((t[0], t[2]),0.5, color = 'blue', alpha = 0.8, fill = False, zorder = 10)
        ax2.add_patch(circle)
        circle = plt.Circle((t[0], t[2]),0.5, color = 'blue', alpha = 0.8, fill = False, zorder = 10)
        ax3.add_patch(circle)

        circle = plt.Circle((t[0], t[2]),0.5, color = 'blue', alpha = 0.8, fill = False, zorder = 10)
        ax_gaze.add_patch(circle)


    ax_yaw.set_xlim(240,300)
    ax_yaw.set_ylim(-60,60)

    ax_gaze.set_xlabel("x-coord (m)")
    ax_gaze.set_ylabel("y-coord (m)")


    plt.figure(fig0.number)
    plt.tight_layout()
    plt.savefig(path + f"traj_lowdpi.jpg", dpi = 200)
    plt.savefig(path + f"traj.png", dpi = 500)
    plt.savefig(path + f"traj.pdf")

    plt.figure(fig2.number)
    plt.tight_layout()
    plt.savefig(path + f"gaze.jpg", dpi = 200)
    plt.savefig(path + f"gaze.png", dpi = 500)
    plt.savefig(path + f"gaze.pdf")


    fig1, axes = plt.subplots(2,5, sharey = 'row', sharex = 'row', figsize=(12, 7))
    fig1.suptitle(title)
    global trial_data
    trial_data = np.array(trial_data)
    args = np.argsort(trial_data[:,0])
    
    percentiles = [0.10, 0.25, 0.5, 0.75, 0.90][::-1]
    percentile_indices = [args[int((len(trial_data) - 1) * p)] for p in percentiles]

    for i in range(len(percentile_indices)):
        p = percentile_indices[i]
        #trial_data.append([ra0, time(inds), fy(inds), fy2(inds), trajr[:,0], trajr[:,1],  traj[:,0], traj[:,1], part, n])
        #yawrates.append([time(ind[c3][0]), ra0, np.sum(c3), time(ind), fy(ind), fy2(ind), x,y,x2,y2, distr - 50, dist - 50])

        axes[0,i].plot(trial_data[p][1], trial_data[p][2], '-g', label = 'Human', linewidth = 1.5, alpha = 0.5)
        axes[0,i].plot(trial_data[p][1], trial_data[p][3], '-', color = 'purple',label = 'Model', linewidth =  1.5, alpha = 0.5)

        axes[1,i].plot(trial_data[p][4], trial_data[p][5], '-g', linewidth =  1.5, alpha = 0.5)
        axes[1,i].plot(trial_data[p][6], trial_data[p][7], '-', color = 'purple',  linewidth =  1.5, alpha = 0.5)

        axes[0,i].set_xlabel("Time (s)")
        axes[1,i].set_xlabel("x-coord (m)")

        #axes[1,i].axhline(-1.75, color = 'black', ls = '--')
        #axes[1,i].axhline(1.75, color = 'black', ls = '--')


        part = int(trial_data[p][-2])
        trial = trial_data[p][-1]
        axes[0,i].set_title(f'Participant: {part - 500}, Trial: {trial}\n{int(percentiles[i]*100)}th percentile', fontsize = 10)


    axes[0,0].set_ylabel("Yawrate (°/s)")
    #axes[0,1].set_title('Yaw rate 50th per')
    #axes[0,2].set_title('Yaw rate 90th per')
    axes[0,0].legend(loc = 'upper left')
    axes[0,2].set_ylim(-40, 40)
    axes[1,0].set_ylim(30, 65)
    axes[1,0].set_xlim(20, 30)

    for i, t in enumerate(obs):
        t = list(t)
        for ax in axes[1,:]:
            circle = plt.Circle((t[0], t[2]),0.5, color = 'blue', alpha = 0.8, fill = False, zorder = 10)
            ax.add_patch(circle)
            ax.plot(inneredge[:,0], inneredge[:,1], '-k', linewidth = 0.2)
            ax.plot(outeredge[:,0], outeredge[:,1], '-k', linewidth = 0.2)
            ax.set_aspect(0.5)



    #axes[1,0].invert_xaxis()
    #axes[1,1].invert_xaxis()
    #axes[1,2].invert_xaxis()
    #axes[2,0].set_ylabel("y-coord (m)")
    #axes[3,0].set_ylabel("y-coord (m)")
    axes[1,0].set_ylabel("y-coord (m)")
    #axes[1,0].set_xlabel("y-coord (m)")
    #axes[1,1].axis('off')

    plt.figure(fig1.number)
    plt.tight_layout()

    #plt.savefig(path + f"yawrates2_lowdip.jpg", dpi = 200)
    #plt.savefig(path + f"yawrates2.png", dpi = 500)
    plt.savefig(path + f"yawrates2.pdf")


    #x = lines[p75].get_data()[0]  - lines[p75].get_data()[0][0]

def cmap(p):
    l = ['#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4', '#f032e6', '#bcf60c', '#fabebe',  '#9a6324', '#800000', '#000075', '#aaffc3', '#808000', '#ffd8b1',  '#808080', '#000000']
    return l[int(p - 501)]

def zero_globals():
    global pos_corr, heatx, heaty, heatxR, heatyR, heatLaneX, heatLaneY, trial_data, gazeX, gazeY, wp_histograms
    pos_corr =  [[],[],[]]
    heatx = []
    heaty = []
    heatxR = []
    heatyR = []
    heatLaneX = []
    heatLaneY = []
    trial_data = []
    wp_histograms = [[],[],[]]

    gazeX = []
    gazeY = []

def performance(mean_diff_sim, yaw_rates, participant, axes, plot = True):
    ax0, ax1, ax2, ax3, ax_yaw, ax_gaze = axes
    mean_diff_sim = np.array(mean_diff_sim)
    yaw_rates = np.array(yaw_rates)
    int_range = [22000 - 16500, 24000 - 16500,  25999 - 16500]
    #print(mean_diff_sim)
    scatter = [[],[],[]]
    total = 0
    total_off = 0

    yaw1 = []
    yaw2 = []
    results = []
    ex_part = 0
    mean_diff_real = true_traj
    if participant >= 500:
        c = (mean_diff_real[:,-1] == participant)
        mean_diff_real = mean_diff_real[c]
        mean_diff_sim  = mean_diff_sim[c]
        yaw_rates = yaw_rates[c]

    for part in np.unique(mean_diff_real[:,-1]):
        c = (mean_diff_real[:,-1] == part)

        n = 0
        part_r = np.zeros((len(mean_path),2))
        part_s = np.zeros((len(mean_path),2))

        for r in np.unique(mean_diff_sim[c,0]):
            if r == -100:
                continue
            c = (mean_diff_real[:,0] == r) & (np.isfinite(mean_diff_real[:,0])) & (mean_diff_real[:,-1] == part)
            
            min_i, max_i = np.min(mean_diff_real[c,1]), min(np.max(mean_diff_real[c,1]), np.max(mean_diff_sim[c,1]))
            #print(max_i, min_i, np.sum(c))
            inds = np.arange(3000, 10500)

            n += 1

            if np.sum(c) == 0:
                print("AAAA")
                continue

            cc = c & (mean_diff_sim[:,1] <= 10500)
            g = mean_diff_sim[cc,-3] 
            if np.sum(g > -2)/len(g) > 0.10:
                #print("aaaa")
                ex_part += 1
                continue
            sign =np.sign(mean_diff_real[c,3][0])

        
            fr = scipy.interpolate.interp1d(mean_diff_real[c,1], mean_diff_real[c,3:5], axis = 0, fill_value = np.nan, bounds_error = False)
            trajr = fr(inds)
            f = scipy.interpolate.interp1d(mean_diff_sim[c,1], mean_diff_sim[c,3:5], axis = 0, fill_value = np.nan, bounds_error = False)
            traj = f(inds)
            results += np.linalg.norm(trajr - traj, axis = 1).tolist()

            #if plot: ax0.plot(fr(inds)[:,0], fr(inds)[:,1], '-', color = cmap(part), alpha = 0.5)    
            if plot: ax0.plot(fr(inds)[:,0], fr(inds)[:,1], '-', color = 'green', alpha = 0.3, linewidth = 0.5)    
            if plot: ax2.plot(f(inds)[:,0], f(inds)[:,1], '-', color = 'purple', alpha = 0.3, linewidth = 0.5)

            #if plot: ax2.plot(f(int_range)[:,0], f(int_range)[:,1], 'ok',  alpha = 1, linewidth = 0.5)


            wp2 = [24.25, 0.1, 36.0]
            wp1 = [25.75, 0.1, 44.0]
            wp0 = [24.25, 0.1, 52.0]
          
            wp_histograms[0].append([fr(int_range[0])[0] - wp0[0],f(int_range[0])[0] - wp0[0]])
            wp_histograms[1].append([fr(int_range[1])[0] - wp1[0],f(int_range[1])[0] - wp1[0] ])
            wp_histograms[2].append([fr(int_range[2])[0] - wp2[0],f(int_range[2])[0] - wp2[0]])

            scatter[0].append([fr(int_range[0])[0],f(int_range[0])[0]])
            scatter[1].append([fr(int_range[1])[0],f(int_range[1])[0]])
            scatter[2].append([fr(int_range[2])[0],f(int_range[2])[0]])
            #ax0.plot(fr(int_range[2])[0], fr(int_range[2])[1], '.k')
            #ax2.plot(f(int_range[2])[0], f(int_range[2])[1], '.k')
            
            time = scipy.interpolate.interp1d(mean_diff_real[c,1], mean_diff_real[c,2], axis = 0, bounds_error = False, fill_value = np.nan)


            fy = scipy.interpolate.interp1d(mean_diff_real[c,1], yaw_rates[c,0], axis = 0, bounds_error = False, fill_value = np.nan)
            yaw1 += fy(inds).tolist()

            if plot: ax_yaw.plot(time(inds) + r*6, fy(inds), '-k', alpha = 0.5)
            
            fy2 = scipy.interpolate.interp1d(mean_diff_sim[c,1], yaw_rates[c,1], axis = 0, bounds_error = False, fill_value = np.nan)
            yaw2 += fy2(inds).tolist()
            if plot: ax_yaw.plot(time(inds) + r*6, fy2(inds), '-', color = cmap(part), alpha = 0.5)

            c3 = np.isfinite(fy(inds)) &  np.isfinite(fy2(inds))
            ra0,p0 = scipy.stats.pearsonr(fy(inds)[c3], fy2(inds)[c3])

            dist =  abs(traj[:,0] - 25)
            total += len(inds)
            total_off += np.sum(dist > 1.5)

            #print(np.sum(dist > 1.5)/len(inds))
            global gazeX, gazeY
            #fg = scipy.interpolate.interp1d(mean_diff_real[c,1], yaw_rates[c,1], axis = 0, bounds_error = False, kind = 'nearest',  fill_value = np.nan)

            gazeX += mean_diff_real[c,-7].tolist()
            gazeY += mean_diff_real[c,-6].tolist()
            #ax_gaze.plot(sign*mean_diff_real[c,-7], sign*mean_diff_real[c,-6], '.r')

            
            global trial_data

            trial_data.append([ra0, time(inds), fy(inds), fy2(inds), trajr[:,0], trajr[:,1],  traj[:,0], traj[:,1], part, n])
            #plt.plot(time(ind) + r*6, f(ind), '-b', alpha = 0.5)


        if plot: ax1.plot(part_r[:,0]/n, part_r[:,1]/n, '-', color = cmap(part), alpha = 1, linewidth = 1)    
        if plot: ax3.plot(part_s[:,0]/n, part_s[:,1]/n, '-', color = cmap(part), alpha =1, linewidth = 1)


        fr = scipy.interpolate.interp1d(indices, part_r/n, axis = 0, fill_value = 'extrapolate', kind = 'nearest', bounds_error = False)
        f = scipy.interpolate.interp1d(indices, part_s/n, axis = 0, fill_value = 'extrapolate', kind = 'nearest', bounds_error = False)

        pos_corr[0].append([fr(int_range[0])[0],f(int_range[0])[0]])
        pos_corr[1].append([fr(int_range[1])[0],f(int_range[1])[0]])
        pos_corr[2].append([fr(int_range[2])[0],f(int_range[2])[0]])

    yaw1 = np.array(yaw1)
    yaw2 = np.array(yaw2)
    #print(yaw1, yaw2)
    c = (np.isfinite(yaw2)) & (np.isfinite(yaw1))

    #print(len(yaw1[c]), len(yaw2[c]))
    #print(scipy.stats.spearmanr(yaw1[c],yaw2[c]))
    yaw_r, yawp = scipy.stats.pearsonr(yaw1[c],yaw2[c])
    #print("corr: ", yaw_r)
    scatter = np.array(scatter)
    ra0,p0 = scipy.stats.pearsonr(scatter[0][:,0], scatter[0][:,1])
    ra1,p1 = scipy.stats.pearsonr(scatter[1][:,0], scatter[1][:,1])
    ra2,p2 = scipy.stats.pearsonr(scatter[2][:,0], scatter[2][:,1])
    #print("means", ra0, ra1, ra2)
    c = np.isfinite(mean_diff_sim[:,5]) & (mean_diff_real[:,1] > 1) #& (mean_diff_sim[:,5] < 50)

    r0 = np.nanmean(results)
    r1 = np.nanmedian(results)
    secres = [r0, np.nanstd(results), (ra0),(ra1), (ra2), yaw_r, total_off/total*100, 0, ex_part]
    return [r0, r1, np.arctanh(ra0), np.arctanh(ra1), np.arctanh(ra2), np.arctanh(yaw_r), 0, total_off/total*100, 0], secres

def heatmap(x,y, fig, ax, bins = 100, lims = []):
    #fig, ax = plt.subplots()
    #return
    cmap = plt.cm.YlOrRd
    cmap.set_under(color='white')
    x = np.array(x)
    y = np.array(y)
    c = (np.isfinite(x)) & (np.isfinite(y))
    x= x[c]
    y = y[c]
    k = gaussian_kde(np.vstack([x, y]))
    #binsx, binsy = 200, 200
    if len(lims) == 4:  
        rat = abs(lims[2] - lims[3]) / abs(lims[0] - lims[1])
        binsy = int(bins*rat)
        xi, yi = np.mgrid[lims[0]:lims[1]:bins*1j, lims[2]:lims[3]:binsy*1j] 
    else: 
        xi, yi = np.mgrid[x.min():x.max():bins*1j, y.min():y.max():bins*1j]  
    zi = k(np.vstack([xi.flatten(), yi. flatten()]))

    area = (np.diff(xi, axis = 0)[0,0] * np.diff(yi, axis = 1)[0,0])
    print(np.sum(zi))
    print("density", np.sum(zi*area))

    pcol = ax.pcolormesh(xi, yi, zi.reshape(xi.shape), alpha = 1.0, cmap=cmap, shading='gouraud', vmin = 0.000001, rasterized = True)
    fig.colorbar(pcol, ax=ax, orientation='vertical', fraction=0.046, pad=0.04)

def preproc_cust(string):
    zero_globals()
    fig0, fig1, fig2, axes = plot(string)
    root = folder
    #root = "./data/"
    #if 'GP' in string:
    #    global true_traj
    #    true_traj = np.load("actual2.npy")
    
    search = string
    if string == 'Real': search = 'Pure_GP'
    for file in os.listdir(root):
        filename = root + os.fsdecode(file)
        if search in filename and not 'yaw' in filename:
            data = np.load(filename)
            yaw = np.load(filename.replace(".npy", "_yaw.npy"))   
    part_res = []
    part_res2 = []
    #r = performance(data, yaw, -1, axes)
    #part_res.append(r)
    for participant in np.unique(data[:,-1]):
        #if participant in [6]: #[3,4,5,6,16]:
        #    continue
        r, r2 = performance(data, yaw, participant, axes, string)
        part_res.append(r)
        part_res2.append(r2)
        #fin_plots(fig0, fig1, fig2, axes)
    if True: #"GP" in string:
        for i in range(3):
            wp = np.array(wp_histograms[i])
            #print(wp, wp.shape)
            if 'Pure' in string: 
                #wp_grid[i].hist(wp[:,0], bins = 10, color = 'green', alpha = 0.5, density = True, histtype='stepfilled')
                k = gaussian_kde(wp[:,0], bw_method = 0.2)
                s,e = np.min(wp[:,0])  - 0.5, np.max(wp[:,0]) + 0.5
                wp_grid[i].plot(np.linspace(s,e,200), k(np.linspace(s,e,200)), '-', color = 'green', label = 'Human')

                #wp_grid[i].hist(wp[:,0], bins = 10, color = 'green', alpha = 1, density = True, histtype='step', label = 'Human')
                k = gaussian_kde(wp[:,1], bw_method = 0.2)
                s,e = np.min(wp[:,1]) - 0.5, np.max(wp[:,1]) + 0.5



                wp_grid[i].plot(np.linspace(s,e,200), k(np.linspace(s,e,200)), '-', color = (0.5,0,0.8), label = 'Pure pursuit controller', alpha = 0.7)

                wp_grid[i].bar(0, -wp_grid[i].get_ylim()[1]*0.06, width = 1, linewidth = 1, fill = True, color = 'blue', label = 'WP extent', alpha = 0.6, clip_on = False)
                wp_grid[i].set_ylim(bottom = 0)
                wp_grid[i].set_xticks([-1.5, -2, -1, -0.5, 0, 0.5, 1, 1.5, 2])
                wp_grid[i].set_xlim(-1, 1)
                #wp_grid[i].hist(wp[:,1],  bins = 10, color = (0.5,0,0.8), alpha = 1, density = True, histtype='step', label = "Purepursuit controller")
                #wp_grid[i].hist(wp[:,1],  bins = 10, color = (0.5,0,1), alpha = 0.5, density = True, histtype='stepfilled')
            else: 
                k = gaussian_kde(wp[:,1], bw_method = 0.2)
                s,e = np.min(wp[:,1]) - 0.5, np.max(wp[:,1]) + 0.5
                wp_grid[i].plot(np.linspace(s,e,200), k(np.linspace(s,e,200)), '-', color = (0.8,0,0.5), label = 'Proportional controller', alpha = 0.7)
                wp_grid[i].set_xlim(-1, 1)
                #wp_grid[i].bar(0, wp_grid[i].get_ylim()[1]*0.1, width = 1, linewidth = 2, fill = False, edgecolor = 'blue')
                #wp_grid[i].hist(wp[:,1],  bins = 10, color = (0.8,0,0.5), alpha = 1, density = True, histtype='step', label = "Proportional controller")
                #wp_grid[i].hist(wp[:,1],  bins = 10, color = (1,0,0.5), alpha = 0.5, density = True, histtype='stepfilled')

            wp_grid[i].set_title(f"Waypoint {i+1}", fontsize = 8)
            wp_grid[i].set_ylabel("Density", fontsize = 8)
            wp_grid[i].tick_params(axis='both', which='major', labelsize=6)
            wp_grid[i].tick_params(axis='both', which='minor', labelsize=6)


    wp_grid[0].legend(fontsize = 6, bbox_to_anchor=(0.7, 1.05))
    wp_grid[2].set_xlabel("Hor-deviation from WP (m)", fontsize = 8)
    #heatmap(heatxR, heatyR, fig4, axes[6])
    #heatmap(heatx, heaty, fig4, axes[7])
    if string == "Pure_GP":
        #axes[4].plot(heatLaneX, heatLaneY, 'b.')
        #heatmap(heatLaneX, heatLaneY, fig3, axes[4])
        global gazeX, gazeY
        gazeX = np.array(gazeX)
        gazeY = np.array(gazeY)
        c = (gazeX < 30) & (gazeX > 20) & (gazeY < 65) & (gazeY > 30)
        #axes[6].plot(gazeX, gazeY, '.r', alpha = 0.01)
        heatmap(gazeX[c], gazeY[c], fig2, axes[-1], lims = [20, 30, 30, 65], bins = 100)


    fin_plots(fig0, fig1, fig2,  axes, string)
    #plt.show()
    return np.array(part_res), np.array(part_res2)


       #[ 1.1   0.59  0.03 -0.02  0.01  0.8   0.19]
       #[ 1.25  1.01 -0.13  0.03  0.07  0.69  0.19]



import pandas as pd
if __name__ == '__main__':
    full = []
    #p0 = preproc_cust("sw_test.npy")
    #print(np.mean(p0, axis = 0))
    #plt.show(block = False)
    group = []
    part_table = []
    c = -4
    #mean - 4, median - 5, pos1 - 6, pos2 - 7, pos - 8, yaw_r - 9

    #part_res = preproc_cust("Real")
    #print(np.round(part_res, 2))
    #print(np.round(np.nanmean(part_res, axis = 0), 3))
    #plt.show()
    part_res, res2 = preproc_cust("Pure_GP")
    part_table.append(res2)
    group.append(part_res[:,c])
    #print(part_res)
    print(np.round(part_res, 2))
    p0 = np.round(np.nanmean(part_res, axis = 0), 3)
    print(p0)
    p = np.nanmean(part_res, axis = 0)
    p = (np.round(np.tanh(p[2:]), 3))
    print(p)
    p0[2:6] = p[0:4]
    full.append(p0)
    #plt.show()
    part_res, res2 = preproc_cust("Prop_GP")
    part_table.append(res2)
    group.append(part_res[:,c])
    print(np.round(part_res, 2))
    p0 = np.round(np.nanmean(part_res, axis = 0), 3)
    print(p0)
    p = np.nanmean(part_res, axis = 0)
    p = (np.round(np.tanh(p[2:]), 3))
    print(p)
    p0[2:6] = p[0:4]
    full.append(p0)
    #plt.show()

    part_res, res2 = preproc_cust("Pure_WP")
    part_table.append(res2)
    group.append(part_res[:,c])
    print(np.round(part_res, 2))
    p0 = np.round(np.nanmean(part_res, axis = 0), 3)
    print(p0)
    p = np.nanmean(part_res, axis = 0)
    p = (np.round(np.tanh(p[2:]), 3))
    print(p)
    p0[2:6] = p[0:4]
    full.append(p0)
    #plt.show()

    part_res, res2 = preproc_cust("Prop_WP")
    part_table.append(res2)
    group.append(part_res[:,c])
    print(np.round(part_res, 2))
    p0 = np.round(np.nanmean(part_res, axis = 0), 3)
    print(p0)
    p = np.nanmean(part_res, axis = 0)
    p = (np.round(np.tanh(p[2:]), 3))
    print(p)
    p0[2:6] = p[0:4]
    full.append(p0)

    full = np.array(full)
    df = pd.DataFrame(full, columns = ['mean','median','corr0', 'corr1', 'corr2', 'yaw', 'fail', 'off', 'excluded'])

    j = 0
    for tab in part_table:
        tab = np.array(tab)
        df_p = pd.DataFrame(tab, columns = ['Mean', 'SD', 'Corr0', 'Corr1', 'Cor2', 'Yaw', 'Prop', 'Failed', 'Excluded'])
        df_p.to_csv(f'part{j}.csv', float_format='%.4f', index = False)
        j += 1

    df.to_csv('results2.csv', float_format='%.3f')

    print(group, np.tanh(np.mean(group, axis = 1)))
    res, p = scipy.stats.ttest_rel(group[0], group[1])
    print(res, p, np.round(res, 2), np.round(p, 2))
    print(scipy.stats.f_oneway(group[0], group[1], group[2], group[3]))

    plt.show()




    """
    asdsad
    #preproc_pooled(4)

    #adsasd
    part_res, part_res2  = preproc("WPure")
    #print(part_res)

    wp_prop = np.array(part_res['WProp'])
    wp_pure = np.array(part_res['WPure'])
    gp_pure = np.array(part_res['GP'])
    gp_prop = np.array(part_res['GProp'])

    p = wp_prop
    print(np.min(p, axis = 0), np.max(p, axis = 0), np.mean(p, axis = 0))

    c = 1
    print(np.mean(gp_pure[:,c]), np.mean(gp_prop[:,c]))
    print(np.sum(gp_prop[:,c] < gp_pure[:,c]))
    print(scipy.stats.ttest_rel(gp_pure[:,c], gp_prop[:,c]))
    """


