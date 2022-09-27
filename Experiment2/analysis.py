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
import matplotlib.patches as patches


warnings.filterwarnings("ignore")

true_traj = np.load("./non_model_data/actual_traj.npy")

mean_path = np.load("./non_model_data/path.npy")
#mean_path[:,0] = scipy.ndimage.gaussian_filter(mean_path[:,0], 50)
#mean_path[:,1] = scipy.ndimage.gaussian_filter(mean_path[:,1], 50)
#inneredge = np.array(track.TrackMaker(10000, 25-1.5)[0])
#outeredge = np.array(track.TrackMaker(10000, 25+1.5)[0])
indices = np.arange(20800)

def rotate(px, py, angle, ox, oy):

    qx = ox + np.cos(angle) * (px - ox) - np.sin(angle) * (py - oy)
    qy = oy + np.sin(angle) * (px - ox) + np.cos(angle) * (py - oy)
    return qx, qy


def road(axes, centrelane = True, turns = 20):
    xr = []
    yr = []

    xr_i = []
    yr_i = []

    xr_o = []
    yr_o = []

    s = 0
    last_x, last_y = -50,0
    for i in range(turns):
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

        if turns == 1:
            road = np.load("./non_model_data/path.npy")
            waypoints = road[::130]
            for ax in axes:
                for t in waypoints[0:11]:
                    #print(t)
                    circle = plt.Circle((t[0], t[1]),0.7, color = 'blue', alpha = 0.8, fill = False, zorder=10)
                    ax.add_patch(circle)

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

    for ax in axes:
        if centrelane: 
            ax.plot(xr - 800, yr, '--k')
            ax.plot(xr_i - 800, yr_i, 'k-')
            ax.plot(xr_o - 800, yr_o, 'k-')
        else:
            ax.plot(xr_i - 800, yr_i, 'k--')
            ax.plot(xr_o - 800, yr_o, 'k--')

comp_fig, comp_axes = None, None


def plot(title = ""):

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


    fig0, axes = plt.subplots(2,1, sharex = True, sharey = True, figsize=(9, 12))
    fig0.suptitle(title)
    ax1, ax3 = axes[0], axes[1]

    fig1, axes = plt.subplots(2,2, sharex = True, sharey = True, figsize=(10, 10))
    ax0, ax2 = axes[0,0], axes[0,1]


    arrow = patches.FancyArrowPatch((mean_path[0,0] + 10, mean_path[0,1]), (mean_path[260,0] + 10, mean_path[260,1]),
                             connectionstyle="arc3,rad=.14",  color="k", arrowstyle = "Simple, tail_width=0.5, head_width=4, head_length=8")
    ax0.add_patch(arrow)
    arrow = patches.FancyArrowPatch((mean_path[0,0] + 10, mean_path[0,1]), (mean_path[260,0] + 10, mean_path[260,1]),
                             connectionstyle="arc3,rad=.14",  color="k", arrowstyle = "Simple, tail_width=0.5, head_width=4, head_length=8")
    ax2.add_patch(arrow)

    arrow = patches.FancyArrowPatch((mean_path[0,0] + 10, mean_path[0,1]), (mean_path[260,0] + 10, mean_path[260,1]),
                             connectionstyle="arc3,rad=.14",  color="k", arrowstyle = "Simple, tail_width=0.5, head_width=4, head_length=8")
    axes[1,0  ].add_patch(arrow)

    axes[1,1].axis('off')

    if "Pure" in title:
        global comp_fig, comp_axes
        comp_fig = fig1
        comp_axes = axes
    else:
        fig1 = comp_fig
        axes = comp_axes
        ax2 = axes[1,0  ]

    #ax1.plot(middle[:,0], middle[:,1], '-k', linewidth = 0.2)
    #ax0.plot(inneredge[:,0], inneredge[:,1], '-k', linewidth = 0.2)
    #ax0.plot(outeredge[:,0], outeredge[:,1], '-k', linewidth = 0.2)

    #ax1.plot(inneredge[:,0], inneredge[:,1], '-k', linewidth = 0.2)
    #ax1.plot(outeredge[:,0], outeredge[:,1], '-k', linewidth = 0.2)

    #ax2.plot(inneredge[:,0], inneredge[:,1], '-k', linewidth = 0.2)
    #ax2.plot(outeredge[:,0], outeredge[:,1], '-k', linewidth = 0.2)

    #ax3.plot(inneredge[:,0], inneredge[:,1], '-k', linewidth = 0.2)
    #ax3.plot(outeredge[:,0], outeredge[:,1], '-k', linewidth = 0.2)

    fig2, ax_yaw = plt.subplots(1,1, sharex = True, sharey = True, figsize = (8,4))
    fig2.suptitle(title)
    """
    obs = track.getObstacles(1)
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
    """

    fig3, ax4 = plt.subplots(1, sharex = True, sharey = True, figsize=(12, 9))
    fig3.suptitle(title)


    fig4, ax5 = plt.subplots(1, sharex = True, sharey = True, figsize=(9, 6))
    #fig4.suptitle(title)
    ax5, ax6 = ax5, ax5

    return fig0, fig1, fig2, fig3, fig4, (ax0, ax1, ax2, ax3, ax4, ax_yaw, ax5, ax6)

def fin_plots(fig0, fig1, fig2, fig3, fig4, axes, string):
    ax0, ax1, ax2, ax3, ax4, ax_yaw, ax5, ax6 = axes

    title = ""
    if 'Prop' in string:
        title += 'Proportional controller with '
    else:
        title += 'Pure pursuit controller with '
    if 'G' in string:
        title += 'gaze input'
    else:
        title += "waypoint input"

    ax4.axhline(-1.75, color = 'black')
    ax4.axhline(1.75, color = 'black')
    road([ax0, ax2, ax5], centrelane = False, turns = 1)
    road([ax1, ax3])
    ax0.set_title('Human trajectories', fontsize = 10)
    ax0.set_xlim(-855, -774)
    ax0.set_ylim(-60, 1)
    ax0.set_xlabel("x-coord (m)")
    ax0.set_ylabel("y-coord (m)")
    ax0.set_aspect('equal')

    ax1.set_title('Real trajectories (part-means)')
    ax1.set_xlim(-850, -690)
    ax1.set_ylim(-85, 0)
    ax1.set_aspect('equal')

    ax2.set_title('Model trajectories\n' + title, fontsize = 10)
    ax2.set_xlim(-855, -774)
    ax2.set_ylim(-60, 1)
    ax2.set_xlabel("x-coord (m)")
    ax2.set_ylabel("y-coord (m)")
    ax2.set_aspect('equal')

    ax3.set_title('Model trajectories (part-means)')
    ax3.set_xlim(-850, -690)
    ax3.set_ylim(-85, 0)
    ax3.set_aspect('equal')

    ax4.set_ylim(-8, 8)
    ax4.set_xlabel("Track pos (m)")
    ax4.set_ylabel("Lane pos (m)")
    #ax4.set_aspect('equal')


    #aspect = 0.7
    #ax0.set_aspect(aspect)
    #ax1.set_aspect(aspect)
    #ax2.set_aspect(aspect)
    #ax3.set_aspect(aspect)

    #ax0.plot(mean_path[:,0], mean_path[:,1], '-k', linewidth = 10)
    #ax1.plot(mean_path[:,0], mean_path[:,1], '-k', linewidth = 10)
    #ax2.plot(mean_path[:,0], mean_path[:,1], '-k', linewidth = 10)
    #ax3.plot(mean_path[:,0], mean_path[:,1], '-k', linewidth = 10)

    #ax_yaw.set_xlim(240,300)
    #ax_yaw.set_ylim(-60,60)

    ax5.set_title('Gaze distribution in experiment 2')



    arrow = patches.FancyArrowPatch((mean_path[0,0] + 10, mean_path[0,1]), (mean_path[260,0] + 10, mean_path[260,1]),
                             connectionstyle="arc3,rad=.14",  color="k", arrowstyle = "Simple, tail_width=0.5, head_width=4, head_length=8")
    ax5.add_patch(arrow)
    ax5.set_xlim(-855, -774)
    ax5.set_ylim(-60, 1)
    ax5.set_xlabel("x-coord (m)")
    ax5.set_ylabel("y-coord (m)")
    ax5.set_aspect('equal')

    #ax6.set_title('Real trajectories')
    #ax6.set_xlim(-850, -690)
    #ax6.set_ylim(-85, 0)
    #ax6.set_aspect('equal')

    plt.figure(fig0.number)
    plt.tight_layout()
    path = "./figures/" + string + "_"


    string = title

    plt.savefig(path + f"part_means_lowdpi.jpg", dpi = 200)
    plt.savefig(path + f"part_means.png", dpi = 500)
    plt.savefig(path + f"part_means.pdf")

    plt.figure(fig1.number)
    plt.tight_layout()
    plt.savefig(path + f"traj_lowdpi.jpg", dpi = 200)
    plt.savefig(path + f"traj.png", dpi = 500)
    plt.savefig(path + f"traj.pdf")

    plt.figure(fig2.number)
    plt.tight_layout()
    plt.savefig(path + f"yaw_lowdpi.jpg", dpi = 200)
    plt.savefig(path + f"yaw.png", dpi = 500)
    plt.savefig(path + f"yaw.pdf")

    plt.figure(fig3.number)
    plt.tight_layout()
    plt.savefig(path + f"offset_lowdpi.jpg", dpi = 200)
    plt.savefig(path + f"offset.png", dpi = 500)
    plt.savefig(path + f"offset.pdf")

    plt.figure(fig4.number)
    plt.tight_layout()
    plt.savefig(path + f"gaze_lowdpi.jpg", dpi = 200)
    plt.savefig(path + f"gaze.png", dpi = 500)
    plt.savefig(path + f"gaze.pdf")

    fig1, axes = plt.subplots(2,5, sharey = 'row', sharex = 'row', figsize=(12, 10), gridspec_kw={'height_ratios': [1, 3]})
    fig1.suptitle(string)
    global yawrates
    yawrates = np.array(yawrates)
    lines = list(ax_yaw.get_lines())

    #yawrates[:,4] = np.rad2deg(yawrates[:,4])
    #yawrates[:,5] = np.rad2deg(yawrates[:,5])
    args = np.argsort(yawrates[:,1])
    
    percentiles = [0.10, 0.25, 0.5, 0.75, 0.90][::-1]
    percentile_indices = [args[int((len(yawrates) - 1) * p)] for p in percentiles]

    #print(yawrates[p25,1], np.percentile(yawrates[:,1], 25))
    #print(yawrates[p50,1], np.percentile(yawrates[:,1], 50))
    #print(yawrates[p75,1], np.percentile(yawrates[:,1], 75))

    #axes[0,0].plot(yawrates[p25][3], np.rad2deg(yawrates[p25][4]), '-b')
    #axes[0,0].plot(yawrates[p25][3], np.rad2deg(yawrates[p25][5]), '-g')

    #axes[0,0].set_xlabel("Time (s)")
    #axes[0,0].plot(lines[p25].get_data()[0], lines[p25].get_data()[1], '-g')
    #axes[0,0].plot(lines[p25].get_data()[0], lines[p25].get_data()[1], '-g')
    #axes[0,0].plot(lines[p25 + 1].get_data()[0], lines[p25 + 1].get_data()[1], '-b')

    #axes[0,1].plot(yawrates[p50][3], np.rad2deg(yawrates[p50][4]), '-b', label = 'Real')
    #axes[0,1].plot(yawrates[p50][3], np.rad2deg(yawrates[p50][5]), '-g', label = 'Simulated')
    #axes[0,1].set_xlabel("Time (s)")
    #axes[0,1].plot(lines[p50].get_data()[0], lines[p50].get_data()[1], '-g')
    #axes[0,1].plot(lines[p50 + 1].get_data()[0], lines[p50 + 1].get_data()[1], '-b')

    #axes[0,2].plot(yawrates[p75][3], np.rad2deg(yawrates[p75][4]), '-b', label = 'Real')
    #axes[0,2].plot(yawrates[p75][3], np.rad2deg(yawrates[p75][5]), '-g', label = 'Simulated')
    #axes[0,2].set_xlabel("Time (s)")
    #axes[0,2].set_ylabel("Yawrate (deg/s)")


    #road(axes[2,:], False)
    #road(axes[3,:], False)
    
    for i in range(len(percentile_indices)):
        p = percentile_indices[i]

        #yawrates.append([time(ind[c3][0]), ra0, np.sum(c3), time(ind), fy(ind), fy2(ind), x,y,x2,y2, distr - 50, dist - 50])

        axes[0,i].plot(yawrates[p][3], np.rad2deg(yawrates[p][4]), '-g', label = 'Human', linewidth = 1.5, alpha = 0.5)
        axes[0,i].plot(yawrates[p][3], np.rad2deg(yawrates[p][5]), '-', color = 'purple', label = 'Model', linewidth =  1.5, alpha = 0.5)

        axes[1,i].plot(yawrates[p][10], yawrates[p][3], '-g', linewidth =  1.5, alpha = 0.5)
        axes[1,i].plot(yawrates[p][11], yawrates[p][3], '-', color = 'purple', linewidth =  1.5, alpha = 0.5)

        #axes[2,i].plot(yawrates[p][6], yawrates[p][7], '-b', linewidth = 1)
        #axes[2,i].plot(yawrates[p][8], yawrates[p][9], '-g', linewidth = 1)
        axes[0,i].set_xlabel("Time (s)")
        axes[1,i].set_xlabel("Lane position (m)")

        for t in yawrates[p][3][::1300]:
            axes[1,i].axhline(t, color = 'grey', ls = '--', alpha = 0.6, linewidth = 0.75)

        axes[1,i].axvline(-1.75, color = 'black', ls = '--')
        axes[1,i].axvline(1.75, color = 'black', ls = '--')
        axes[1,i].invert_yaxis()
        #axes[2,i].set_xlim(-850, -690)
        #axes[2,i].set_ylim(-85, 0)
        #axes[2,i].set_aspect('equal')
        #axes[2,i].set_xlabel("x-coord (m)")

        #axes[3,i].plot(yawrates[p][12], yawrates[p][13], '.r', alpha = 0.01)
        #axes[3,i].set_xlim(-850, -690)
        #axes[3,i].set_ylim(-85, 0)
        #axes[3,i].set_aspect('equal')
        #axes[3,i].set_xlabel("x-coord (m)")

        #axes[2,i].plot(yawrates[p][12], yawrates[p][13], '.r', alpha = 0.01)
        #axes[2,i].set_xlim(-850, -690)
        #axes[2,i].set_ylim(-85, 0)
        #axes[2,i].set_aspect('equal')
        #axes[2,i].set_xlabel("x-coord (m)")

        part = int(yawrates[p][-2])
        trial = yawrates[p][-1]
        axes[0,i].set_title(f'Participant: {part}, Trial: {trial}\n{int(percentiles[i]*100)}th percentile', fontsize = 10)
    #axes[0,1].set_title('Participant: {p}, Trial: {t}\n25th percentile', fontsize = 10)
    #axes[0,2].set_title('Participant: {p}, Trial: {t}\n50th percentile', fontsize = 10)
    #axes[0,3].set_title('Participant: {p}, Trial: {t}\n75th percentile', fontsize = 10)
    #axes[0,4].set_title('Participant: {p}, Trial: {t}\n90th percentile', fontsize = 10)
    axes[1,0].set_ylabel("Time (s)")
    axes[0,0].set_ylabel("Yawrate (°/s)")
    #axes[0,1].set_title('Yaw rate 50th per')
    #axes[0,2].set_title('Yaw rate 90th per')
    axes[0,0].legend(loc = 'upper left')
    axes[0,2].set_ylim(-35, 35)
    axes[1,0].set_xlim(-3, 3)
    #axes[2,0].set_ylabel("y-coord (m)")
    #axes[3,0].set_ylabel("y-coord (m)")

    #axes[1,1].axis('off')

    plt.figure(fig1.number)
    plt.tight_layout()
    plt.savefig(path + f"yawrates2_lowdip.jpg", dpi = 200)
    plt.savefig(path + f"yawrates2.png", dpi = 500)
    plt.savefig(path + f"yawrates2.pdf")

    fig, ax = plt.subplots(1, figsize=(6, 6))
    #ax.plot(heatx, heaty, '.r', alpha = 0.01)
    #heatmap(heatx, heaty, fig, ax)

    for l in yawrates:
        #axes[2,i].plot(l[6], l[7], color = 'green', linewidth = 1)
        #ax.plot(l[3], l[11], color = 'green', linewidth = 1, alpha = 0.1)
        ax.plot(l[8], l[9], color = 'green', linewidth = 1, alpha = 0.1)
    road([ax], False)
    #ax.axhline(-1.75, color = 'black')
    #ax.axhline(1.75, color = 'black')
    ax.set_xlim(-850, -690)
    ax.set_ylim(-85, 0)
    ax.set_xlabel("X-coord (m)")
    ax.set_ylabel("Y-coord (m)")

    plt.figure(fig.number)
    plt.tight_layout()
    plt.savefig(path + f"roads_lowdip.jpg", dpi = 200)
    plt.savefig(path + f"roads.png", dpi = 500)
    plt.savefig(path + f"roads.pdf")



    fig, ax = plt.subplots(1, figsize=(6, 6))
    #ax.plot(heatx, heaty, '.r', alpha = 0.01)
    #heatmap(heatx, heaty, fig, ax)
    for l in yawrates:
        #axes[2,i].plot(l[6], l[7], color = 'green', linewidth = 1)
        ax.plot(l[3], l[11], color = 'green', linewidth = 1, alpha = 0.1)

    ax.axhline(-1.75, color = 'black')
    ax.axhline(1.75, color = 'black')
    ax.set_ylim(-8, 8)
    ax.set_xlabel("Track pos (m)")
    ax.set_ylabel("Lane pos (m)")

    plt.figure(fig.number)
    plt.tight_layout()
    plt.savefig(path + f"lanes_lowdip.jpg", dpi = 200)
    plt.savefig(path + f"lanes.png", dpi = 500)
    plt.savefig(path + f"lanes.pdf")




    #x = lines[p75].get_data()[0]  - lines[p75].get_data()[0][0]

def cmap(p):
    l = ['#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4', '#f032e6', '#bcf60c', '#fabebe',  '#9a6324', '#800000', '#000075', '#aaffc3', '#808000', '#ffd8b1',  '#808080', '#000000']
    return l[int(p - 1)]

def zero_globals():
    global pos_corr, heatx, heaty, heatxR, heatyR, heatLaneX, heatLaneY, yawrates, gazeX, gazeY
    pos_corr =  [[],[],[]]
    heatx = []
    heaty = []
    heatxR = []
    heatyR = []
    heatLaneX = []
    heatLaneY = []
    yawrates = []

    gazeX = []
    gazeY = []


def performance(mean_diff_sim, yaw_rates, participant, axes, cond, plot = True):
    ax0, ax1, ax2, ax3, ax4, ax_yaw, ax5, ax6 = axes

    mean_diff_sim = np.array(mean_diff_sim)
    yaw_rates = np.array(yaw_rates)
    int_range = [7000, 14000, 19000]
    #print(mean_diff_sim)
    scatter = [[],[],[]]

    yaw1 = []
    yaw2 = []
    mean_diff_real = true_traj
    origs = np.load("./non_model_data/origs.npy")
    corigs = np.load("./non_model_data/circle_origs.npy")

    if participant >= 0:
        c = (mean_diff_real[:,-1] == participant)
        mean_diff_real = mean_diff_real[c]
        mean_diff_sim  = mean_diff_sim[c]
        yaw_rates = yaw_rates[c]
    if cond == 'Real':
        mean_diff_sim[:,1] = np.array(mean_diff_real[:,1])
        mean_diff_sim[:,3] = np.array(mean_diff_real[:,3])
        mean_diff_sim[:,4] = np.array(mean_diff_real[:,4])
        yaw_rates[:,1] = -yaw_rates[:,0]

    sign, signr = np.sign(mean_diff_sim[:,4]), np.sign(mean_diff_real[:,4])
    mean_diff_sim[:,4] *= -sign #-np.abs(mean_diff_sim[:,4])
    mean_diff_real[:,4] *= -signr #-np.abs(mean_diff_real[:,4])

    #mean_diff_sim[:,6] = -mean_diff_sim[:,3]
    mean_diff_sim[:,7] *= -signr

    total = 0
    total_off = 0

    trials = 0
    fails = 0
    
    part_r = np.zeros((len(indices),2))
    part_s = np.zeros((len(indices),2))

    part_rn = np.zeros((len(indices),2))
    part_sn = np.zeros((len(indices),2))      

    distance = np.array(mean_diff_sim[:,5]) 
    mean_diff_sim[:,5]  = np.nan

    trial_off_prop = []
    results = []    

    part = participant
    ex_part = 0
    trial = 0
    for r in np.unique(mean_diff_sim[:,0]):
        if r == -100:
            continue

        c = (mean_diff_real[:,0] == r) & (np.isfinite(mean_diff_real[:,0])) & (mean_diff_real[:,1] < 20000)

        g = mean_diff_sim[c,-3]
        g_h = g > -3
        #g_h = np.diff(g_h)
        s = 0
        maxs = 0
        #for t in g_h:
        #    if t: s += 1
        #    else:
        #        if s > maxs:
        #            maxs = s 
        #        s = 0
        #if maxs > 120:
        #    continue

 #& (mean_diff_sim[:,1] < 20800)  & (mean_diff_real[:,1] < 20800)
        #print(np.max(mean_diff_real[c,1]), np.max(mean_diff_sim[c,1]))
        #print(np.sum(c),mean_diff_sim[:,1])
        if np.sum(c) < 8000:
            continue

        trial += 1
    
        if np.sum(g > -2)/len(g) > 0.10:
            #print("aaaa")
            ex_part += 1
            continue

        total += np.sum(c)


        total_off += np.sum(distance[c] > 1.75)
        trials += 1
        fail = False
        if np.sum(distance[c] > 10) > 0:
            fails += 1
            fail = True
            #continue
            mean_diff_real[:,4] = -np.abs(mean_diff_real[:,4])
        c = c & (np.isfinite(mean_diff_sim[:,4])) #& (np.isfinite(mean_diff_sim[:,3])) 
        diff = np.where(np.diff(mean_diff_sim[c,1]) < 0)[0]

        if len(diff) > 0:
            #print(mean_diff_sim[c][diff[0], 1])
            #print(mean_diff_real[c][diff[0], 1])
            #print(diff, diff[0], len(c))
            c[diff[0] + np.where(c)[0][0]: ] = False
            if not fail:
                #print("BBBB")
                fails += 1
                fail = True
        if not fail:
            #print("AAAAA")
            trial_off_prop.append(np.sum(distance[c] > 1.75)/len(distance[c]))
        #if np.sum(c) == 0:
        #    continue

        #a = np.argsort(mean_diff_sim[c,1])
        #mean_diff_sim[c] = mean_diff_sim[c][a]


        fr = scipy.interpolate.interp1d(mean_diff_real[c,1], mean_diff_real[c,3:5], axis = 0, fill_value = np.nan, bounds_error = False)

        ind = np.arange(1300, 1300*15)
        orig_inds = (ind//1300).astype(int)
        dat =  fr(ind) - origs[orig_inds]
        c2 = np.isfinite(dat[:,0])
        part_r[ind[c2]] += dat[c2]
        part_rn[ind[c2]] += 1


        f = scipy.interpolate.interp1d(mean_diff_sim[c,1], mean_diff_sim[c,3:5], axis = 0,  fill_value = np.nan,  bounds_error = False)
        fg = scipy.interpolate.interp1d(mean_diff_sim[c,1], mean_diff_sim[c,6:8], axis = 0, bounds_error = False, fill_value = np.nan, kind = 'nearest')

        dat =  f(ind) - origs[orig_inds]
        c2 = np.isfinite(dat[:,0])
        part_s[ind[c2]] += dat[c2]
        part_sn[ind[c2]] += 1


        o_inds = (ind // 1300).astype(int)
        o = origs[o_inds]
        locs = np.where(np.diff(o_inds) != 0)[0]
        #print(np.diff(np.unique(o[:,1])))
        #o[o_inds%2 != 0, 1] -= np.diff(np.unique(o[:,1]))[0]/2
        #o[o_inds%2 != 0, 0] += np.diff(np.unique(o[:,0]))[0]/2

        x = fr(ind)[:,0] - o[:,0]
        y = fr(ind)[:,1]  - o[:,1]
        #y[o_inds%2 != 0] *= -1
        global heatxR, heatyR
        global heatx, heaty
        #heatxR += x.tolist()
        #heatyR += (fr(mean_diff_sim[c,1])[:,1]  - o[:,1]).tolist()

        


        x2 = f(ind)[:,0] - o[:,0]

        circles = corigs[o_inds]
        dist = (f(ind)[:,0] - circles[:,0] + 800)**2  +  (f(ind)[:,1]  - circles[:,1])**2
        dist = dist**0.5

        distr= (fr(ind)[:,0] - circles[:,0] + 800)**2  +  (fr(ind)[:,1]  - circles[:,1])**2
        distr = distr**0.5


        #print(dist)
        x = np.sin((ind%1300/1300*-np.pi*2/3 - np.pi*0.5))*distr - 800
        y = np.cos((ind%1300/1300*-np.pi*2/3 - np.pi*0.5))*distr
        x[locs] = np.nan
        if plot: ax0.plot(x, y, '-', color = 'green', alpha = 0.05, linewidth = 0.5)  
        #if plot: ax0.plot(x[0], y[0], 'o', color = cmap(part), alpha = 0.1)  

        x = np.sin((ind%1300/1300*-np.pi*2/3 - np.pi*0.5))*dist - 800
        y = np.cos((ind%1300/1300*-np.pi*2/3 - np.pi*0.5))*dist
        x[locs] = np.nan
        if plot: ax2.plot(x, y, '-', color = 'purple', alpha = 0.05, linewidth = 0.5)  
        #if plot: ax2.plot(x[0], y[0], 'o', color = cmap(part), alpha = 0.1)  


        dist2 = (fg(ind)[:,0] - circles[:,0] + 800)**2  +  (fg(ind)[:,1]  - circles[:,1])**2
        dist2 = dist2**0.5
        heatx +=  (ind*0.08061566110620978).tolist()
        heaty += (dist2 - 50).tolist()

        global heatLaneX, heatLaneY
        #ax4.plot(mean_diff_sim[c,1]*0.08061566110620978, dist - 50, '-', alpha = 0.2, color = 'purple')

        heatLaneX += (ind*0.08061566110620978).tolist()
        heatLaneY += (dist - 50).tolist()


        #ax4.plot(circles[:,0] - 800, circles[:,1], 'ob')
        #ax4.plot(f(mean_diff_sim[c,1])[:,0], f(mean_diff_sim[c,1])[:,1], '-b')



        #heatx += x.tolist()
        #heaty += (f(mean_diff_sim[c,1])[:,1] - o[:,1]).tolist()
        x2[locs] = np.nan
        y2 = f(ind)[:,1] - o[:,1]
        #if plot: ax2.plot(x2, y2, '-', color = cmap(part), alpha = 0.1)
        #if plot: ax2.plot(x2[0], y2[0], 'o', color = cmap(part), alpha = 0.1)
        #plt.show()
        results += (np.linalg.norm(fr(ind) - f(ind), axis = 1)).tolist()
        #print("a", np.nanmean(np.linalg.norm(fr(mean_diff_real[c,1]) - f(mean_diff_real[c,1]), axis = 1)))
        #print(np.nanmean(np.linalg.norm(fr(mean_diff_sim[c,1]) - f(mean_diff_sim[c,1]), axis = 1)))

        samp = [0, ind[int(len(ind)/2)], -1]

        scatter[0].append((dist[samp[0]], distr[samp[0]]))
        scatter[1].append((dist[samp[1]], distr[samp[1]]))
        scatter[2].append((dist[samp[2]], distr[samp[2]]))
    
        #scatter[0].append([fr(int_range[0])[0],f(int_range[0])[0]])
        #scatter[1].append([fr(int_range[1])[0],f(int_range[1])[0]])
        #scatter[2].append([fr(int_range[2])[0],f(int_range[2])[0]])

        time = scipy.interpolate.interp1d(mean_diff_real[c,1], mean_diff_real[c,2], axis = 0, bounds_error = False, fill_value = np.nan)

        fy = scipy.interpolate.interp1d(mean_diff_real[c,1], -yaw_rates[c,0], axis = 0, bounds_error = False, fill_value = np.nan)
        yaw1 += fy(ind).tolist()

        if plot: ax_yaw.plot(time(ind) + r*300, fy(ind), '-k', alpha = 0.5)

        fy2 = scipy.interpolate.interp1d(mean_diff_sim[c,1], yaw_rates[c,1], axis = 0, bounds_error = False, fill_value = np.nan)
        yaw2 += fy2(ind).tolist()
        if plot: ax_yaw.plot(time(ind) + r*300, fy2(ind), '-', color = cmap(part), alpha = 0.5)
        
        c3 = np.isfinite(fy(ind)) & np.isfinite(fy2(ind)) 
        if np.sum(c3) > 0: ra0,p0 = scipy.stats.pearsonr(fy(ind[c3]), fy2(ind[c3]))

        #gaze = fg(ind)
        #gaze[:,0] -= o[:,0]
        #gaze[:,1] -= o[:,1]


        #fg = scipy.interpolate.interp1d(mean_diff_sim[c,1], mean_diff_sim[c,6:8], axis = 0, bounds_error = False, fill_value = np.nan)
        global gazeX, gazeY
        #o_inds = (ind // 1300).astype(int)
        #o = origs[o_inds]
        gaze = fg(mean_diff_sim[c,1])
        #f_gind = scipy.interpolate.interp1d(mean_diff_sim[c,1], mean_diff_sim[c,-5], axis = 0, bounds_error = False, fill_value = np.nan, kind = 'nearest')
        #print(np.max(mean_diff_sim[c,-5]), np.min(mean_diff_sim[c,-5]))
        #nind = f_gind(ind)
        #print(np.max(f_gind(ind)), np.min(f_gind(ind)))
        #if np.sum(~np.isfinite(nind)) > 0:
        #    plt.figure()
        #    plt.plot(gaze[:,0], gaze[:,1], 'or')
        #    print(fail)
        #    plt.show()
        #print(np.max(f_gind(ind)), np.min(f_gind(ind)))
        o_inds = (mean_diff_sim[c,-5] // 1300).astype(int)
        o = origs[o_inds]

        circles = corigs[o_inds]
        distg = (gaze[:,0] - circles[:,0] + 800)**2  +  (gaze[:,1]  - circles[:,1])**2
        distg = distg**0.5
        i = mean_diff_sim[c,-5]
        #print(dist, mean_diff_sim[c,-5])
        gaze[:,0] = np.sin((i%1300/1300*-np.pi*2/3 - np.pi*0.5))*distg - 800
        gaze[:,1] = np.cos((i%1300/1300*-np.pi*2/3 - np.pi*0.5))*distg
        #print(len(x), len(y), gaze.shape)
        #plt.figure()
        #plt.plot(x, y, 'or')
        #plt.show()
        #gaze[:,0] -= o[:,0]
        #gaze[:,1] -= o[:,1]
        #print(len(inds), len(time(inds)))
        if np.sum(c3) > 0: yawrates.append([time(ind[c3][0]), ra0, np.sum(c3), time(ind), fy(ind), fy2(ind), x,y,x2,y2, distr - 50, dist - 50, gaze[:,0], gaze[:,1], part, trial])

        gazeX += gaze[:,0].tolist()
        gazeY += gaze[:,1].tolist()




        #yawrates.append([time(ind[c3][0]), ra0, np.sum(c3)])            

        #plt.plot(time(ind) + r*6, f(ind), '-b', alpha = 0.5)

    n = 2600
    for i in range(1,8):
         part_r[:n,0] += part_r[i*n:(i+1)*n, 0]
         part_rn[:n,0] += part_rn[i*n:(i+1)*n, 0]
         part_r[:n,1] += part_r[i*n:(i+1)*n, 1]
         part_rn[:n,1] += part_rn[i*n:(i+1)*n, 1]

         part_s[:n,0] += part_s[i*n:(i+1)*n, 0]
         part_sn[:n,0] += part_sn[i*n:(i+1)*n, 0]
         part_s[:n,1] += part_s[i*n:(i+1)*n, 1]
         part_sn[:n,1] += part_sn[i*n:(i+1)*n, 1]


    part_r = part_r[:n]  
    part_rn = part_rn[:n]       
    part_s = part_s[:n]  
    part_sn = part_sn[:n]         

    part_r[:,0] /= part_rn[:,0]
    part_r[:,1] /= part_rn[:,1]

    part_s[:,0] /= part_sn[:,0]
    part_s[:,1] /= part_sn[:,1]



    c1, c2  = np.isfinite(part_r[:,0]) & (part_rn[:,1] > 0),  np.isfinite(part_s[:,0]) & (part_sn[:,1] > 0)
    #print(trials)
    if  np.sum(c1) == 0:
        return [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],    [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]
    
    #locs = np.where(np.diff(indices//1300) != 0)[0]
    #part_r[locs,0] = np.nan
    #part_s[locs,0] = np.nan

    if plot: ax1.plot(part_r[c1,0], part_r[c1,1], '-', color = cmap(part), alpha = 1, linewidth = 1)    
    if plot: ax3.plot(part_s[c2,0], part_s[c2,1], '-', color = cmap(part), alpha = 1, linewidth = 1)
    

    fr = scipy.interpolate.interp1d(indices[:n][c1], part_r[c1], axis = 0, fill_value = np.nan, bounds_error = False)
    f = scipy.interpolate.interp1d(indices[:n][c2], part_s[c2], axis = 0, fill_value = np.nan, bounds_error = False)
    #ax1.plot(fr(int_range[0])[0],fr(int_range[0])[1], 'o', color = cmap(part), alpha = 1)
    #ax3.plot(f(int_range[0])[0],f(int_range[0])[1], 'o', color = cmap(part), alpha = 1)
    a,b,c = int(n*0.25), int(n*0.5), int(n*0.75)
    pos_corr[0].append([fr(a)[0],f(a)[0]])
    pos_corr[1].append([fr(b)[0],f(b)[0]])
    pos_corr[2].append([fr(c)[0],f(c)[0]])

    yaw1 = np.array(yaw1)
    yaw2 = np.array(yaw2)
    #print(yaw1, yaw2)
    c = (np.isfinite(yaw2)) & (np.isfinite(yaw1))

    #print(len(yaw1[c]), len(yaw2[c]))
    #print(scipy.stats.spearmanr(yaw1[c],yaw2[c]))
    yaw_r, yawp = scipy.stats.pearsonr(yaw1[c],yaw2[c])
    #print("corr: ", yaw_r)
    scatter = np.array(scatter)
    c = np.isfinite(scatter[0][:,0]) & np.isfinite(scatter[0][:,1])
    if np.sum(c) > 1:
        ra0,p0 = scipy.stats.pearsonr(scatter[0][c,0], scatter[0][c,1])
    else: ra0, p0 = np.nan, np.nan
    c = np.isfinite(scatter[1][:,0]) & np.isfinite(scatter[1][:,1])
    if np.sum(c) > 1:
        ra1,p1 = scipy.stats.pearsonr(scatter[1][c,0], scatter[1][c,1])
    else: ra1, p1 = np.nan, np.nan  
    c = np.isfinite(scatter[2][:,0]) & np.isfinite(scatter[2][:,1])
    if np.sum(c) > 1:
        ra2,p2 = scipy.stats.pearsonr(scatter[2][c,0], scatter[2][c,1])
    else: ra2, p2 = np.nan, np.nan  


    results = np.array(results)

    r0 = np.nanmean(results)
    r1 = np.nanmedian(results)

    mean_fail_prop = np.nanmean(trial_off_prop)

    res = [r0, r1, np.arctanh(ra0), np.arctanh(ra1), np.arctanh(ra2), np.arctanh(yaw_r), fails/trials, total_off/total*100, ex_part]
    secres = [r0, np.nanstd(results), (ra0),(ra1), (ra2), yaw_r, total_off/total*100, fails, ex_part]
    return res, secres

def heatmap(x,y, fig, ax, bins = 100, lims = []):
    return
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

    pcol = ax.pcolormesh(xi, yi, zi.reshape(xi.shape), alpha = 1.0, cmap=cmap, edgecolors = 'green', linewidth = 3, shading='gouraud', vmin = 0.000001)
    fig.colorbar(pcol, ax=ax, orientation='vertical', fraction=0.046, pad=0.04)

def preproc_cust(string):
    zero_globals()
    fig0, fig1, fig2, fig3, fig4, axes = plot(string)
    root = "./model_data/"
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
        #plt.show()

    #heatmap(heatxR, heatyR, fig4, axes[6])
    #heatmap(heatx, heaty, fig4, axes[7])
    if string == "Pure_GP":
        #axes[4].plot(heatLaneX, heatLaneY, 'b.')
        heatmap(heatLaneX, heatLaneY, fig3, axes[4])
        global gazeX, gazeY
        gazeX = np.array(gazeX)
        gazeY = np.array(gazeY)
        c = (gazeX < -764) & (gazeX > -860) & (gazeY < 5) & (gazeY > -75)
        #axes[6].plot(gazeX, gazeY, '.r', alpha = 0.01)
        heatmap(gazeX[c], gazeY[c], fig4, axes[6], lims = [-860, -764, -75, 5], bins = 100)

    fin_plots(fig0, fig1, fig2, fig3, fig4, axes, string)
    #plt.close("all")
    #plt.figure()
    #plt.hist(heatLaneY, bins = 50)
    #plt.show()
    pos = np.array(pos_corr)
    #print(pos)
    if (len(pos[0]) > 2):
        c = np.isfinite(pos[0][:,1])
        ra0,p0 = scipy.stats.pearsonr(pos[0][c,0], pos[0][c,1])
        c = np.isfinite(pos[1][:,1])
        ra1,p1 = scipy.stats.pearsonr(pos[1][c,0], pos[1][c,1])
        c = np.isfinite(pos[2][:,1])
        ra2,p2 = scipy.stats.pearsonr(pos[2][c,0], pos[2][c,1])
        #print(pos[0][:,0], pos[0][:,1])
        print(ra0, ra1, ra2)

    return np.array(part_res), np.array(part_res2)


       #[ 1.1   0.59  0.03 -0.02  0.01  0.8   0.19]
       #[ 1.25  1.01 -0.13  0.03  0.07  0.69  0.19]



import pandas as pd
if __name__ == '__main__':
    full = []
    #p0 = preproc_cust("sw_test.npy")
    #print(np.mean(p0, axis = 0))
    #plt.show(block = False)

    #mean - 4, median - 5, pos1 - 6, pos2 - 7, pos - 8, yaw_r - 9
    c = 0
    group = []
    part_table = []
    #part_res = preproc_cust("Real")
    #print(np.round(part_res, 2))
    #print(np.round(np.nanmean(part_res, axis = 0), 3))
    #plt.show()
    part_res, res2 = preproc_cust("Pure_GP")
    part_table.append(res2)
    #print(part_res)
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

    df.to_csv('results2.csv', float_format='%.3f')
    #0,np.arctanh(ra0), np.arctanh(ra1), np.arctanh(ra2), yaw_r, total_off/total, fails/trials, ex_part
    j = 0
    for tab in part_table:
        tab = np.array(tab)
        df = pd.DataFrame(tab, columns = ['Mean', "SD", 'Corr0', 'Corr1', 'Cor2', 'Yaw', 'Prop', 'Failed', 'Excluded'])
        df.to_csv(f'part{j}.csv', float_format='%.4f', index = False)
        j += 1
    group = np.array(group)
    group = np.array(group).T
    print(group.shape)
    group = group[np.isfinite(group[:,0])].T
    print(np.mean(group, axis = 1), np.tanh(np.mean(group, axis = 1)))
    print(group.shape)
    res, p = scipy.stats.ttest_rel(group[0], group[1])
    print(res, p, np.round(res, 2), np.round(p, 3))


    print(scipy.stats.f_oneway(group[0], group[1]))
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


